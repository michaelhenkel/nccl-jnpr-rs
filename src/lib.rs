use std::ffi::{c_void, CStr};
use std::io::{Read, Write};
use std::net::{Ipv6Addr, TcpListener, TcpStream};
use std::os::raw::c_char;
use std::ptr::{self, null_mut};
use std::thread::{self, JoinHandle};
use rdma_sys::*;

mod bindings;
use bindings::*;
mod common;
use common::*;
use serde::{Serialize, Deserialize};


use libc::{c_int, size_t};
// struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct SocketComm{
    qpn: u32,
    qp_idx: u32,
    total_qps: u32,
    gid_subnet_ids: u64,
    gid_interface_ids: u64,
    psn: u32,
    stage: u8,
    md_remote_address: u64,
    md_rkey: u32,
    md_length: u32,
}

pub struct ListenComm{
    rx: std::sync::mpsc::Receiver<SocketCommand>,
    listen_comm_handle: JoinHandle<()>,
    stage: *mut Stages,
}

#[derive(Debug)]
enum Stages{
    Listen,
    Connect,
    Accept
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct NcclNetSocketHandle {
    ipv6_address: [u8; 16],
    port: u16,
    stage: u8,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct RemoteHandle{
    qpn: u32,
    gid_subnet_ids: u64,
    gid_interface_ids: u64,
    psn: u32,
}

pub struct QpList{
    qps: Vec<Qp>,
    context: IbvContext,
    pd: ProtectionDomain,
    gid_table: GidTable,
    port: u16,
    send_recv: u8,
    remote_md_mr: MemoryRegion,
    local_md: MetaDataWrapper,
    remote_md: MetaData,
}


impl QpList{
    fn new(device_name: String, port: u16, send_recv: u8) -> anyhow::Result<QpList, CustomError>{
        let gid_table = generate_gid_table(Some(device_name)).unwrap();
        let context = gid_table.context.clone();
        let pd = ProtectionDomain::new(context.clone())?;
        Ok(QpList{
            qps: Vec::new(),
            context,
            pd,
            gid_table,
            port,
            send_recv,
            remote_md_mr: MemoryRegion(null_mut()),
            local_md: MetaDataWrapper(null_mut()),
            remote_md: MetaData::default(),
        })
    }
    fn create_qps(&mut self) -> anyhow::Result<(), CustomError>{
        let mut qp_idx = 0;
        for (_v6, gid_entry) in &self.gid_table.v6_table{
            /*
            let event_channel = create_event_channel(self.context.ibv_context()).unwrap();
            let cq = create_create_completion_queue(self.context.ibv_context(), event_channel.event_channel());
            let qp = create_queue_pair(self.pd.pd(), cq);
            */
            let qp = Qp::new(self.pd.clone(), self.context.clone(), gid_entry.clone())?;
            self.qps.push(qp);
            qp_idx += 1;
        }

        Ok(())
    }
    fn add_qp(&mut self, idx: usize) -> anyhow::Result<(), CustomError>{
        if self.gid_table.v6_table.len() - 1 < idx{
            return Err(CustomError::new("Not enough QPs".to_string(), -1));
        }
        let v6_table_vec: Vec<GidEntry> = self.gid_table.v6_table.values().cloned().collect();
        let gid_entry = v6_table_vec.get(idx).unwrap();
        let qp = Qp::new(self.pd.clone(), self.context.clone(), gid_entry.clone())?;
        self.qps.push(qp);
        Ok(())
    }
}

#[derive(Clone)]
pub struct Qp{
    qp: *mut ibv_qp,
    cq: *mut ibv_cq,
    event_channel: *mut ibv_comp_channel,
    psn: u32,
    gid_entry: GidEntry,
}

unsafe impl Send for Qp{}
unsafe impl Sync for Qp{}

impl Qp{

    fn new(pd: ProtectionDomain, context: IbvContext, gid_entry: GidEntry) -> anyhow::Result<Qp, CustomError>{
        
        let event_channel = create_event_channel(context.ibv_context()).unwrap();
        let cq = create_create_completion_queue(context.ibv_context(), event_channel.event_channel());
        let qp = create_queue_pair(pd.pd(), cq);
        
        /*
        let event_channel = EventChannel::new(context.clone())?;
        let cq = CompletionQueue::new(context.clone(), event_channel.clone())?;
        let qp = QueuePair::new(pd.clone(), cq.clone())?;
        */
        set_qp_init_state(qp.clone(), gid_entry.port)?;
        let psn = rand::random::<u32>() & 0xffffff;

        Ok(Qp{
            qp,
            cq,
            event_channel: event_channel.event_channel(),
            psn,
            gid_entry,
        })
    }
    
}

extern "C" fn plugin_init(logFunction: ncclDebugLogger_t) -> ncclResult_t {
    println!("{} plugin_init called", get_hostname());
    0
}
extern "C" fn plugin_devices(ndev: *mut c_int) -> ncclResult_t {
    println!("{} plugin_devices called", get_hostname());
    unsafe { ndev.write(1) };
    
    0
}
extern "C" fn plugin_get_properties(dev: c_int, props: *mut ncclNetProperties_v8_t) -> ncclResult_t {
    println!("{} plugin_get_properties called", get_hostname());
    match get_device_properties("mlx5_1".to_string()){
        Ok(_props) => {
            unsafe {
                std::ptr::write(props, _props);
            }
            return 0;

        },
        Err(e) => {
            return e.code().try_into().unwrap();
        }
    }
}
extern "C" fn plugin_listen(_dev: c_int, handle: *mut c_void, listen_comm: *mut *mut c_void) -> ncclResult_t {
    println!("{} plugin_listen called", get_hostname());
    let listen_address = get_listen_address("mlx5_1".to_string()).unwrap();
    let listen_address = if let Some(listen_address) = listen_address{
        listen_address
    }else{
        println!("listen_address is None");
        return 1;
    };
    let port = portpicker::pick_unused_port().unwrap();
    let listen_address_bytes = listen_address.octets();
    let listen_address = format!("[{}]:{}", listen_address.to_string(), port);
    let (tx, rx) = std::sync::mpsc::channel();
    let listen_comm_handle = socket_listener(listen_address, tx);
    let stage = Stages::Listen;
    let boxed_stage = Box::new(stage);
    let stage_ptr = Box::into_raw(boxed_stage);
    
    let listen_comm_object = ListenComm{
        rx,
        listen_comm_handle,
        stage: stage_ptr,
    };
    let boxed_handle = Box::into_raw(Box::new(listen_comm_object));
    if !listen_comm.is_null() {
        unsafe {
            *listen_comm = boxed_handle as *mut c_void;
        }
    }
    if !handle.is_null(){
        println!("{} plugin_listen handle is not null", get_hostname());
        let net_sock_handle = handle as *mut NcclNetSocketHandle;
        let stage = unsafe { (*net_sock_handle).stage };
        println!("{} plugin_listen stage: {}", get_hostname(), stage);
        unsafe {
            (*net_sock_handle).ipv6_address = listen_address_bytes;
            (*net_sock_handle).port = port;
            (*net_sock_handle).stage += 1;
        }
    } else {
        println!("{} plugin_listen handle is null", get_hostname());
    }
    /*
    let listen_handle = NcclNetSocketHandle {
        ipv6_address: listen_address_bytes,
        port,
    };
    unsafe {
        let handle = handle as *mut NcclNetSocketHandle;
        *handle = listen_handle;
    }
    */
    0
}
extern "C" fn plugin_connect(_dev: c_int, handle: *mut c_void, send_comm: *mut *mut c_void, _send_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    println!("{} plugin_connect called", get_hostname());
    let handle = handle as *mut NcclNetSocketHandle;
    let port = unsafe { (*handle).port };
    let ipv6_address_bytes = unsafe { (*handle).ipv6_address };
    let listen_address = Ipv6Addr::from(ipv6_address_bytes);
    let send_address = format!("[{}]:{}", listen_address.to_string(), port);
    let mut stream = TcpStream::connect(send_address).unwrap();

    let mut qp_list = QpList::new("mlx5_1".to_string(), port, 0).unwrap();
    qp_list.create_qps().unwrap();
    let num_qps = qp_list.qps.len();
    for (idx, qp) in qp_list.qps.iter().enumerate(){
        let qpn = unsafe { (*qp.qp).qp_num };
        let subnet_id = unsafe { (*qp.gid_entry.gid.ibv_gid()).global.subnet_prefix };
        let interface_id = unsafe { (*qp.gid_entry.gid.ibv_gid()).global.interface_id };
        let psn = qp.psn;
        let socket_comm = SocketComm{
            qpn,
            qp_idx: (idx as u32) +1,
            total_qps: num_qps as u32,
            gid_subnet_ids: subnet_id,
            gid_interface_ids: interface_id,
            psn,
            stage: 0,
            md_remote_address: 0,
            md_rkey: 0,
            md_length: 0,
        };
        let serialized = bincode::serialize(&socket_comm).unwrap();
        stream.write_all(&serialized).unwrap();
        let mut buffer = vec![0; 1024]; // Adjust size if necessary
        stream.read(&mut buffer).unwrap();
        let remote_socket_comm: SocketComm = bincode::deserialize(&buffer).unwrap();
        if remote_socket_comm.stage == 1{
            let remote_subnet_id = remote_socket_comm.gid_subnet_ids;
            let remote_interface_id = remote_socket_comm.gid_interface_ids;
            let subnet_prefix_bytes = remote_subnet_id.to_be_bytes();
            let interface_id_bytes = remote_interface_id.to_be_bytes();
            let subnet_prefix_bytes = subnet_prefix_bytes.iter().rev().cloned().collect::<Vec<u8>>();
            let interface_id_bytes = interface_id_bytes.iter().rev().cloned().collect::<Vec<u8>>();
            let mut raw = [0u8; 16];
            raw[..8].copy_from_slice(&subnet_prefix_bytes);
            raw[8..].copy_from_slice(&interface_id_bytes);
            let remote_gid = ibv_gid{
                raw,
            };
            let remote_qpn = remote_socket_comm.qpn;
            let remote_psn = remote_socket_comm.psn;
            let psn = qp.psn;
            let gidx = qp.gid_entry.gidx;
    
    
            if let Err(e) = connect_qp(qp.qp, remote_gid, remote_qpn, remote_psn, psn, gidx){
                println!("Error connecting QP: {:?}", e);
                return e.code().try_into().unwrap();
            };
        }
    }

    let local_md: MetaData = MetaData::default();
    
    let mr_flags = ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0 as i32 | ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0 as i32 | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0 as i32;
    let mr = unsafe { ibv_reg_mr(qp_list.pd.pd(), &local_md as *const _ as *mut c_void, std::mem::size_of::<MetaData>() as usize, mr_flags) };
    let mr_address = unsafe { (*mr).addr as u64 };
    let mr_rkey = unsafe { (*mr).rkey };
    let mr_length = unsafe { (*mr).length as u32 };
    let mr_lkey = unsafe { (*mr).lkey };
    let local_md = Box::into_raw(Box::new(local_md));

    println!("{} plugin_connect created local_md with: address {} rkey {} lkey {} length {}",get_hostname(), mr_address, mr_rkey, mr_lkey, mr_length);

    let socket_comm = SocketComm{
        qpn: 0,
        qp_idx: 0,
        total_qps: 0,
        gid_subnet_ids: 0,
        gid_interface_ids: 0,
        psn: 0,
        stage: 2,
        md_remote_address: mr_address,
        md_rkey: mr_rkey,
        md_length: mr_length,
    };
    let serialized = bincode::serialize(&socket_comm).unwrap();
    stream.write_all(&serialized).unwrap();

    let mut buffer = vec![0; 1024]; // Adjust size if necessary
    stream.read(&mut buffer).unwrap();
    let remote_socket_comm: SocketComm = bincode::deserialize(&buffer).unwrap();
    if remote_socket_comm.stage == 3{
        println!("{} plugin_connect received remote_md with: address {} rkey {} length {}",get_hostname(), remote_socket_comm.md_remote_address, remote_socket_comm.md_rkey, remote_socket_comm.md_length);
        let remote_md: MetaData = MetaData{
            remote_address: remote_socket_comm.md_remote_address,
            rkey: remote_socket_comm.md_rkey,
            length: remote_socket_comm.md_length,
            lkey: mr_lkey,
        };
        qp_list.remote_md_mr = MemoryRegion(mr);
        qp_list.local_md = MetaDataWrapper(local_md);
        qp_list.remote_md = remote_md;
    }

    let boxed_handle = Box::into_raw(Box::new(qp_list));

    unsafe {
        *send_comm = boxed_handle as *mut c_void;
    }
    println!("{} plugin_connect finished", get_hostname());
    0
}
extern "C" fn plugin_accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void, _recv_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    println!("{} plugin_accept called", get_hostname());

    unsafe {
        let handle = Box::from_raw(listen_comm as *mut ListenComm);
        let rx = handle.rx;
        let request = rx.recv().unwrap();
        match request{
            SocketCommand::Connect { socket_comm_list, qp_list } => {
                for (idx, qp) in qp_list.qps.iter().enumerate(){
                    let remote_socket_comm = socket_comm_list.get(idx).unwrap();
                    let remote_subnet_id = remote_socket_comm.gid_subnet_ids;
                    let remote_interface_id = remote_socket_comm.gid_interface_ids;
                    let subnet_prefix_bytes = remote_subnet_id.to_be_bytes();
                    let interface_id_bytes = remote_interface_id.to_be_bytes();
                    let subnet_prefix_bytes = subnet_prefix_bytes.iter().rev().cloned().collect::<Vec<u8>>();
                    let interface_id_bytes = interface_id_bytes.iter().rev().cloned().collect::<Vec<u8>>();
                    let mut raw = [0u8; 16];
                    raw[..8].copy_from_slice(&subnet_prefix_bytes);
                    raw[8..].copy_from_slice(&interface_id_bytes);
                    let remote_gid = ibv_gid{
                        raw,
                    };
                    if let Err(e) = connect_qp(
                        qp.qp,
                        remote_gid,
                        remote_socket_comm.qpn,
                        remote_socket_comm.psn,
                        qp.psn,
                        qp.gid_entry.gidx,
                    ) {
                        return e.code().try_into().unwrap();
                    }
                }
                let boxed_handle = Box::into_raw(Box::new(qp_list));
                *recv_comm = boxed_handle as *mut c_void;
            }
        }
    }

    //let qp_list = unsafe { Box::from_raw(recv_comm as *mut QpList) };
    //println!("num qps after unpack: {}", qp_list.qps.len());
    println!("{} plugin_accept finished", get_hostname());
    0
}
extern "C" fn plugin_reg_mr(coll_comm: *mut c_void, data: *mut c_void, size: size_t, _type_: c_int, mhandle: *mut *mut c_void) -> ncclResult_t {
    let qp_list = unsafe { Box::from_raw(coll_comm as *mut QpList) };
    println!("{} plugin_reg_mr called, send_recv {}", get_hostname(), qp_list.send_recv);
    let pd = qp_list.pd.clone();
    let access_flags = ibv_access_flags::IBV_ACCESS_REMOTE_WRITE | ibv_access_flags::IBV_ACCESS_LOCAL_WRITE | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
    let mr: *mut ibv_mr = unsafe { ibv_reg_mr(pd.pd(), data, size, access_flags.0 as i32) };
    if mr == null_mut() {
        return 1;
    }
    unsafe {
        let addr = (*mr).addr as u64;
        let length = (*mr).length as u32;
        let lkey = (*mr).lkey;
        println!("{} plugin_reg_mr registered mr with addr: {}, length: {}, lkey: {}",get_hostname(), addr, length, lkey);
    }
    //let boxed_mr = Box::into_raw(Box::new(mr));
    unsafe { *mhandle = mr as *mut c_void };
    Box::into_raw(qp_list); 
    0
}
extern "C" fn plugin_reg_mr_dma_buf(coll_comm: *mut c_void, data: *mut c_void, size: size_t, _type_: c_int, _offset: u64, _fd: c_int, _mhandle: *mut *mut c_void) -> ncclResult_t {
    let qp_list = unsafe { Box::from_raw(coll_comm as *mut QpList) };
    /*
                        let access_flags = ibv_access_flags::IBV_ACCESS_REMOTE_WRITE | ibv_access_flags::IBV_ACCESS_LOCAL_WRITE | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
                    let addr = data.addr();
                    let mr = unsafe { ibv_reg_mr(pd.pd(), addr, memory_region_request.size as usize, access_flags.0 as i32) };
                    if mr == null_mut() {
                        return Err(CustomError::new("Failed to register memory region".to_string(), -1));
                    }
    
     */
    let pd = qp_list.pd.clone();
    let access_flags = ibv_access_flags::IBV_ACCESS_REMOTE_WRITE | ibv_access_flags::IBV_ACCESS_LOCAL_WRITE | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
    let mr = unsafe { ibv_reg_mr(pd.pd(), data, size, access_flags.0 as i32) };
    if mr == null_mut() {
        return 1;
    }
    println!("plugin_reg_mr_dma_buf called");
    0
}
extern "C" fn plugin_dereg_mr(_coll_comm: *mut c_void, _mhandle: *mut c_void) -> ncclResult_t { 
    println!("plugin_dereg_mr called");
    0
}

extern "C" fn plugin_isend(send_comm: *mut c_void, data: *mut c_void, size: c_int, tag: c_int, mhandle: *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    println!("{} plugin_isend called, tag {}, size {}", get_hostname(), tag, size);
    let qp_list = unsafe { Box::from_raw(send_comm as *mut QpList) };
    let md_mr = qp_list.remote_md_mr.clone();
    unsafe {
        let meta_data = (*md_mr.0).addr as *mut MetaData;
        let addrs = (*meta_data).remote_address;
        let rkey = (*meta_data).rkey;
        let length = (*meta_data).length;
        println!("local_md addr: {}, rkey: {}, length: {}", addrs, rkey, length);
    }
    
    unsafe {
        let mhandle = mhandle as *mut ibv_mr;
        let addr = (*mhandle).addr as u64;
        let length = (*mhandle).length as u32;
        let lkey = (*mhandle).lkey;
        println!("{} plugin_isend mhandle addr: {}, length: {}, lkey: {}",get_hostname(), addr, length, lkey);
    }
    

    let remote_md = qp_list.remote_md;
    println!("remote_md: {:#?}", remote_md);
    thread::sleep(std::time::Duration::from_secs(5));
    /*
    let qp_list = unsafe { Box::from_raw(send_comm as *mut QpList) };
    let mr = unsafe { Box::from_raw(mhandle as *mut *mut ibv_mr) };
    let mut wr_list = create_wr(data, size, mr.clone(), qp_list.qps.len() as i32);
    let mut jh_list = Vec::new();
    for (qp_idx, qp) in qp_list.qps.iter().enumerate(){
        let wr = wr_list.remove(qp_idx);
        let wr = IbvSendWr(wr);
        let ec = EventChannel(qp.event_channel);
        let cq = CompletionQueue(qp.cq);
        let qp = QueuePair(qp.qp);
        
        
        let jh = thread::spawn(move || {
            let mut bad_wr: *mut ibv_send_wr = ptr::null_mut();
            let ret = unsafe { ibv_post_send(qp.qp(), wr.send_wr(), &mut bad_wr) };
            if ret != 0 {
                println!("Error posting send: {}", ret);
            }
            let _ret = unsafe { send_complete(cq.cq(), ec.event_channel(), 1, ibv_wc_opcode::IBV_WC_RDMA_WRITE).unwrap() };
        });
        jh_list.push(jh);

    }
    for jh in jh_list{
        jh.join().unwrap();
    }
    Box::into_raw(qp_list);
    Box::into_raw(mr);
    */

    thread::sleep(std::time::Duration::from_secs(5));

    

    0
}
extern "C" fn plugin_irecv(recv_comm: *mut c_void, n: c_int, data: *mut *mut c_void, sizes: *mut c_int, tags: *mut c_int, mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    
    println!("{} plugin_irecv called n: {}",get_hostname(), n);

    let qp_list = unsafe { Box::from_raw(recv_comm as *mut QpList) };
    let md_mr = qp_list.remote_md_mr.clone();
    /*
    unsafe {
        //let x = Box::from_raw(qp_list.local_md.0);
        //let x_addr = x.addr();
        //println!("local_md x addr: {}", x_addr);
        (*qp_list.local_md.0).set_remote_address(666);
        (*qp_list.local_md.0).remote_address = 666;
        let ra = (*qp_list.local_md.0).remote_address;
        println!("local_md remote_address: {}", ra);
        let ptr_addr = qp_list.local_md.0 as *const _ as *mut c_void as u64;
        println!("local_md ptr_addr: {}", ptr_addr as u64);
    }
    */

    let remote_md = qp_list.remote_md.clone();

    
    unsafe {
        let meta_data = (*md_mr.0).addr as *mut MetaData;
        (*meta_data).remote_address = 666;
    }
    
    let md_mr_addr = unsafe { (*md_mr.0).addr as u64 };
    let md_mr_length = unsafe { (*md_mr.0).length as u32 };
    let md_mr_lkey = unsafe { (*md_mr.0).lkey };
    println!("md_mr_addr: {}", md_mr_addr);


    for i in 0..n {
        let tag_item = unsafe { *tags.offset(i as isize) };
        println!("tag_item: {}", tag_item);
        let size_item = unsafe { *sizes.offset(i as isize) };
        println!("size_item: {}", size_item);
        let rkey = remote_md.rkey;
        let remote_address = remote_md.remote_address;
        let mhandle = unsafe { *mhandles.offset(i as isize) };
        unsafe {
            let mhandle = mhandle as *mut ibv_mr;
            let addr = (*mhandle).addr as u64;
            let length = (*mhandle).length as u32;
            let lkey = (*mhandle).lkey;
            println!("{} plugin_irecv mhandle addr: {}, length: {}, lkey: {}",get_hostname(), addr, length, lkey);
        }
        println!("sending to remote_address: {}, rkey: {}", remote_address, rkey);
        //local_md.remote_address = local_md.addr();
        let sge = ibv_sge{
            addr: md_mr_addr,
            length: md_mr_length,
            lkey: md_mr_lkey,
        };
        let sge = Box::new(sge);
        let sge_ptr: *mut ibv_sge = Box::into_raw(sge);
        let mut wr = unsafe { std::mem::zeroed::<ibv_send_wr>() };
        wr.sg_list = sge_ptr;
        wr.num_sge = 1;
        wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
        wr.wr.rdma.remote_addr = remote_address;
        wr.wr.rdma.rkey = rkey;
        wr.wr_id = tag_item as u64;
        wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;
        let mut bad_wr: *mut ibv_send_wr = ptr::null_mut();
        let ret = unsafe { ibv_post_send(qp_list.qps[0].qp, &mut wr, &mut bad_wr) };
        if ret != 0 {
            println!("Error posting send: {}", ret);
        }

    }
    let request = Request{
        id: 0,
    };

    let request_handle = Box::into_raw(Box::new(request));
    unsafe { * _request = request_handle as *mut c_void };
    /*
    let qp_list = unsafe { Box::from_raw(recv_comm as *mut QpList) };
    let mut jh_list = Vec::new();
    for i in 0..n {
        // Access each element using pointer arithmetic
        let data_item = unsafe { *data.offset(i as isize) };
        let size_item = unsafe { *sizes.offset(i as isize) };
        let mhandle_item = unsafe { *mhandles.offset(i as isize) };
        let mr = unsafe { Box::from_raw(mhandle_item as *mut *mut ibv_mr) };
        let wr_list = create_wr(data_item, size_item, mr.clone(), qp_list.qps.len() as i32);
        for (qp_idx, qp) in qp_list.qps.iter().enumerate(){
            println!("wr_list len: {}, qp_list len: {}, qp_idx: {}", wr_list.len(), qp_list.qps.len(), qp_idx);
            let wr = wr_list.get(qp_idx).unwrap();
            let wr = IbvSendWr(wr.clone());
            let ec = EventChannel(qp.event_channel);
            let cq = CompletionQueue(qp.cq);
            let qp = QueuePair(qp.qp);
            
            
            let jh = thread::spawn(move || {
                let mut bad_wr: *mut ibv_send_wr = ptr::null_mut();
                let ret = unsafe { ibv_post_send(qp.qp(), wr.send_wr(), &mut bad_wr) };
                if ret != 0 {
                    println!("Error posting send: {}", ret);
                }
                //let _ret = unsafe { send_complete(cq.cq(), ec.event_channel(), 1, ibv_wc_opcode::IBV_WC_RDMA_WRITE).unwrap() };
            });
            jh_list.push(jh);
    
        }
        Box::into_raw(mr);

    }
    for jh in jh_list{
        jh.join().unwrap();
    }
    */
    //thread::sleep(std::time::Duration::from_secs(1));

    //Box::into_raw(qp_list);
    Box::into_raw(qp_list);
    
    0
}
extern "C" fn plugin_iflush(_recv_comm: *mut c_void, _n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    println!("plugin_iflush called");
    1
}
extern "C" fn plugin_test(_request: *mut c_void, _done: *mut c_int, _size: *mut c_int) -> ncclResult_t { 
    println!("plugin_test called");
    0
}
extern "C" fn plugin_close_send(_send_comm: *mut c_void) -> ncclResult_t { 
    println!("plugin_close_send called");
    0
}
extern "C" fn plugin_close_recv(_recv_comm: *mut c_void) -> ncclResult_t { 
    println!("plugin_close_recv called");
    0
}
extern "C" fn plugin_close_listen(_listen_comm: *mut c_void) -> ncclResult_t { 
    println!("plugin_close_listen called");
    0
}
extern "C" fn plugin_irecv_consumed(_recv_comm: *mut c_void, _n: c_int, _request: *mut c_void) -> ncclResult_t { 
    println!("plugin_irecv_consumed called");
    0
}
extern "C" fn plugin_get_device_mr(comm: *mut c_void, mhandle: *mut c_void, dptr_mhandle: *mut *mut c_void) -> ncclResult_t {
    println!("{} plugin_get_device_mr called", get_hostname());
    let qp_list = unsafe { Box::from_raw(comm as *mut QpList) };
    let col_type = qp_list.send_recv;

    let mr = unsafe { Box::from_raw(mhandle as *mut *mut ibv_mr) };

    let boxed_mr = Box::into_raw(Box::new(mr.clone()));
    unsafe { *dptr_mhandle = boxed_mr as *mut c_void };

    Box::into_raw(mr);
    Box::into_raw(qp_list); 

    0
}

#[no_mangle]
pub static ncclNetPlugin_v8: ncclNet_v8_t = ncclNet_v8_t {
    name: "jnpr\0".as_ptr() as *const c_char,
    init: Some(plugin_init),
    devices: Some(plugin_devices),
    getProperties: Some(plugin_get_properties),
    listen: Some(plugin_listen),
    connect: Some(plugin_connect),
    accept: Some(plugin_accept),
    regMr: Some(plugin_reg_mr),
    regMrDmaBuf: Some(plugin_reg_mr_dma_buf),
    deregMr: Some(plugin_dereg_mr),
    isend: Some(plugin_isend),
    irecv: Some(plugin_irecv),
    iflush: Some(plugin_iflush),
    test: Some(plugin_test),
    closeSend: Some(plugin_close_send),
    closeRecv: Some(plugin_close_recv),
    closeListen: Some(plugin_close_listen),
    getDeviceMr: Some(plugin_get_device_mr),
    irecvConsumed: Some(plugin_irecv_consumed),
};

fn socket_listener(listen_address: String, tx: std::sync::mpsc::Sender<SocketCommand>) -> JoinHandle<()>{
    
    let listen_comm_handle = thread::spawn(move || {
        let listener = TcpListener::bind(listen_address).unwrap();
        let mut qp_list = QpList::new("mlx5_1".to_string(), 0, 1).unwrap();
        let mut remote_socket_comm_list = Vec::new();

        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            loop{
                let mut buffer = vec![0; 1024]; // Adjust size if necessary
                let bytes_read = stream.read(&mut buffer).unwrap();
                if bytes_read == 0 {
                    break;
                }
                let remote_socket_comm: SocketComm = bincode::deserialize(&buffer).unwrap();

                if remote_socket_comm.stage == 0{
                    remote_socket_comm_list.push(remote_socket_comm.clone());
                    let qp_idx = remote_socket_comm.qp_idx - 1;
                    qp_list.add_qp(qp_idx as usize).unwrap();
                    let qp = qp_list.qps.get(qp_idx as usize).unwrap();
                    let qpn = unsafe { (*qp.qp).qp_num };
                    let subnet_id = unsafe { (*qp.gid_entry.gid.ibv_gid()).global.subnet_prefix };
                    let interface_id = unsafe { (*qp.gid_entry.gid.ibv_gid()).global.interface_id };
                    let psn = qp.psn;
                    let socket_comm = SocketComm{
                        qpn,
                        qp_idx,
                        total_qps: qp_list.qps.len() as u32,
                        gid_subnet_ids: subnet_id,
                        gid_interface_ids: interface_id,
                        psn,
                        stage: 1,
                        md_remote_address: 0,
                        md_rkey: 0,
                        md_length: 0,
                    };
                    let serialized = bincode::serialize(&socket_comm).unwrap();
                    stream.write_all(&serialized).unwrap();
                }

                if remote_socket_comm.stage == 2 {
                    let local_md: MetaData = MetaData::default();

                    let mr_flags = ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0 as i32 | ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0 as i32 | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0 as i32;
                    let mr = unsafe { ibv_reg_mr(qp_list.pd.pd(), &local_md as *const _ as *mut c_void, std::mem::size_of::<MetaData>() as usize, mr_flags) };
                    let mr_address = unsafe { (*mr).addr as u64 };
                    let mr_rkey = unsafe { (*mr).rkey };
                    let mr_length = unsafe { (*mr).length as u32 };
                    let mr_lkey = unsafe { (*mr).lkey };
                    let local_md_ptr = &local_md as *const _ as *mut c_void;
                    println!("{} plugin_accept mr_address: {}, ptr_addr: {}",get_hostname(), mr_address, local_md_ptr as u64);
                    
                    println!("{} plugin_accept created local_md with: address {} rkey {} lkey {} length {}",get_hostname(), mr_address, mr_rkey, mr_lkey, mr_length);
                    let local_md = Box::into_raw(Box::new(local_md));

                    let mut remote_md: MetaData = MetaData::default();
                    remote_md.remote_address = remote_socket_comm.md_remote_address;
                    remote_md.rkey = remote_socket_comm.md_rkey;
                    remote_md.length = remote_socket_comm.md_length;
                    remote_md.lkey = mr_lkey;
                    println!("{} plugin_accept received remote_md with: address {} rkey {} length {}",get_hostname(), remote_socket_comm.md_remote_address, remote_socket_comm.md_rkey, remote_socket_comm.md_length);

                    let mut socket_comm = SocketComm::default();
                    socket_comm.md_remote_address = mr_address;
                    socket_comm.md_rkey = mr_rkey;
                    socket_comm.md_length = mr_length;
                    socket_comm.stage = 3;
                    let serialized = bincode::serialize(&socket_comm).unwrap();
                    stream.write_all(&serialized).unwrap();
                    qp_list.remote_md_mr = MemoryRegion(mr);
                    qp_list.local_md = MetaDataWrapper(local_md);
                    qp_list.remote_md = remote_md;
                    break;
                }
            }
            break;
        }
        let socket_command = SocketCommand::Connect { socket_comm_list: remote_socket_comm_list, qp_list};
        tx.send(socket_command).unwrap();

        //qp_list
    });
    listen_comm_handle
}

fn get_hostname() -> String {
    let hostname = hostname::get().unwrap();
    hostname.into_string().unwrap()
}