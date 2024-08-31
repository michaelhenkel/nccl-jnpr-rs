use std::ffi::{c_void, CStr};
use std::fmt::Display;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::os::raw::c_char;
use std::pin::Pin;
use std::ptr::null_mut;
use std::thread;

use ibverbs_rs::receiver::Receiver;
use ibverbs_rs::sender::Sender;
use ibverbs_rs::{print_wr_ids, ControlBufferTrait, Hints, IbvAccessFlags, IbvMr, IbvRecvWr, IbvSendWr, IbvWcOpcode, IbvWrOpcode, LookUpBy, QpMode, SendRecv};
use rdma_sys::*;
use std::sync::Once;
use env_logger::Env;
mod bindings;
use bindings::*;
use serde::{Serialize, Deserialize};

static INIT: Once = Once::new();
pub fn initialize_logger() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(Env::default().default_filter_or("info"))
            //.filter_module("my_library", LevelFilter::Info)
            .init();
    });
}

use libc::{c_int, size_t, stat};

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct NcclNetSocketHandle {
    ipv6_address: [u8; 16],
    ipv4_address: [u8; 4],
    family: u8,
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

enum SenderReceiver{
    Sender{
        sender: Sender,
        state: State,
    },
    Receiver{
        receiver: Receiver,
        state: State,
    }
}

#[derive(Debug, Default)]
struct State{
    id: u32,
    mrs: u32,
    nreqs: u32,
    completed: u32,
    posted: u32,
}

extern "C" fn plugin_init(_log_function: ncclDebugLogger_t) -> ncclResult_t {
    0
}
extern "C" fn plugin_devices(ndev: *mut c_int) -> ncclResult_t {
    unsafe { ndev.write(1) };
    
    0
}
extern "C" fn plugin_get_properties(_dev: c_int, props: *mut ncclNetProperties_v8_t) -> ncclResult_t {
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
    let port = portpicker::pick_unused_port().unwrap();
    let lookup_by = LookUpBy::Name("mlx5_1".to_string());
    let qp_mode = QpMode::Multi;
    let mut receiver = match Receiver::new::<NcclMetadata>(lookup_by, port, qp_mode){
        Ok(receiver) => receiver,
        Err(e) => {
            log::error!("Error creating receiver: {:?}", e);
            return 1;
        }
    };
    if let Err(e) = receiver.create_control_buffer(){
        log::error!("Error creating metadata MR: {:?}", e);
        return 1;
    }

    let hints = Hints::AddressFamily(ibverbs_rs::Family::Inet6);

    if let Err(e) = receiver.listen(hints){
        log::error!("Error listening: {:?}", e);
        return 1;
    }
    let listen_address = receiver.get_listen_address();
    let mut netsock_handler = NcclNetSocketHandle::default();
    match listen_address {
        IpAddr::V4(address) => {
            netsock_handler.ipv4_address = address.octets();
            netsock_handler.family = 4;
        },
        IpAddr::V6(address) => {
            netsock_handler.ipv6_address = address.octets();
            netsock_handler.family = 6;
        }
    }
    netsock_handler.port = port;

    if !handle.is_null() {
        unsafe {
            let handle = handle as *mut NcclNetSocketHandle;
            *handle = netsock_handler;
        }
    }
    let mut state = State::default();
    state.id = rand::random::<u32>();
    let listen_comm_handle = Box::into_raw(Box::new(SenderReceiver::Receiver { receiver, state }));
    if !listen_comm.is_null() {
        unsafe {
            *listen_comm = listen_comm_handle as *mut c_void;
        }
    }
    0
}
extern "C" fn plugin_connect(_dev: c_int, handle: *mut c_void, send_comm: *mut *mut c_void, _send_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    let handle = handle as *mut NcclNetSocketHandle;
    let port = unsafe { (*handle).port };
    let receiver_family = unsafe { (*handle).family };
    let receiver_address = match receiver_family {
        4 => {
            let listen_address_bytes = unsafe { (*handle).ipv4_address };
            IpAddr::V4(Ipv4Addr::from(listen_address_bytes))

        },
        6 => {
            let listen_address_bytes = unsafe { (*handle).ipv6_address };
            IpAddr::V6(Ipv6Addr::from(listen_address_bytes))
        },
        _ => {
            return 1;
        }
    };
    let lookup_by = LookUpBy::Name("mlx5_1".to_string());

    let mut sender = match Sender::new::<NcclMetadata>(
        lookup_by,
        receiver_address,
        port,
        1,
        ibverbs_rs::Family::Inet6,
        QpMode::Multi,
    ){
        Ok(sender) => sender,
        Err(e) => {
            log::error!("Error creating sender: {:?}", e);
            return 1;
        }
    };

    if let Err(e) = sender.create_control_buffer(){
        log::error!("Error creating metadata: {:?}", e);
        return 1;
    }

    if let Err(e) = sender.connect(){
        log::error!("Error connecting: {:?}", e);
        return 1;
    }
    let mut state = State::default();
    state.id = rand::random::<u32>();
    let sender_handle = Box::into_raw(Box::new(SenderReceiver::Sender{
        sender,
        state,
    }));

    unsafe {
        *send_comm = sender_handle as *mut c_void;
    }
    0
}
extern "C" fn plugin_accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void, _recv_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    let mut sender_receiver = unsafe { Box::from_raw(listen_comm as *mut SenderReceiver) };
    if let SenderReceiver::Receiver{ref mut receiver, state: _} = *sender_receiver{
        if let Err(e) = receiver.accept(){
            log::error!("Error accepting: {:?}", e);
            return 1;
        }
        let receiver_handle = Box::into_raw(sender_receiver);
        unsafe {
            *recv_comm = receiver_handle as *mut c_void;
        }
    } else {
        log::error!("Error accepting: {:?}", "Not a receiver");
        return 1;
    }

    0
}
extern "C" fn plugin_reg_mr(coll_comm: *mut c_void, data: *mut c_void, size: size_t, _type_: c_int, mhandle: *mut *mut c_void) -> ncclResult_t {
    let access_flags = IbvAccessFlags::LocalWrite.as_i32() | IbvAccessFlags::RemoteWrite.as_i32() | IbvAccessFlags::RemoteRead.as_i32();
    let mut sender_receiver = unsafe { Box::from_raw(coll_comm as *mut SenderReceiver) };
    let mr = match *sender_receiver{
        SenderReceiver::Sender{ref mut sender, ref mut state} => {
            state.mrs += 1;
            sender.incr_mrs();
            let mr = IbvMr::new(sender.pd.clone(), data, size, access_flags);
            mr
        },
        SenderReceiver::Receiver{ref mut receiver, ref mut state} => {
            state.mrs += 1;
            receiver.incr_mrs();
            let data_addr = data as u64;
            let mr = IbvMr::new(receiver.pd.clone(), data, size, access_flags);
            println!("{} plugin_reg_mr id {} mr addr {}, data addr {}", get_hostname(), state.id, mr.addr(), data_addr);
            mr
        }
    };
    let mr_handle = Box::into_raw(Box::new(mr));
    unsafe { *mhandle = mr_handle as *mut c_void; }
    Box::into_raw(sender_receiver); 
    0
}
extern "C" fn plugin_reg_mr_dma_buf(_coll_comm: *mut c_void, _data: *mut c_void, _size: size_t, _type_: c_int, _offset: u64, _fd: c_int, _mhandle: *mut *mut c_void) -> ncclResult_t {
    0
}
extern "C" fn plugin_dereg_mr(_coll_comm: *mut c_void, _mhandle: *mut c_void) -> ncclResult_t { 
    0
}

extern "C" fn plugin_isend(send_comm: *mut c_void, _data: *mut c_void, _size: c_int, _tag: c_int, _mhandle: *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 

    let mut sender_receiver = unsafe { Box::from_raw(send_comm as *mut SenderReceiver) };
    let (sender, state) = if let SenderReceiver::Sender{ref mut sender, ref mut state} = *sender_receiver{
        (sender, state)
    } else {
        log::error!("Error accepting: {:?}", "Not a sender");
        return 1;
    };

    let read_wr = ibverbs_rs::IbvSendWr::new(
        &sender.in_buffer_mr(),
        sender.out_remote_buffer_addr(),
        sender.out_remote_buffer_rkey(),
        IbvWrOpcode::RdmaRead,
    );
    if let Err(e) = sender.qp_list[0].ibv_post_send(read_wr.as_ptr()){
        log::error!("Error posting read: {:?}", e);
        return 1;
    }
    /*
    if let Err(e) = sender.qp_list[0].complete(1, IbvWcOpcode::RdmaRead, SendRecv::Send){
        log::error!("Error completing: {:?}", e);
        return 1;
    }
    */
    let (remote_addr, remote_rkey) = loop {
        let in_nccl_metadata = NcclMetadata::from(sender.in_buffer_mr());
        let remote_addr = in_nccl_metadata.requests[0].address;
        let remote_rkey = in_nccl_metadata.requests[0].rkey;
        if remote_addr != 0 && remote_rkey != 0 {
            break (remote_addr, remote_rkey);
        }
    };

    let mr = unsafe { Box::from_raw(_mhandle as *mut IbvMr) };
    let send_wr = ibverbs_rs::IbvSendWr::new(
        &mr,
        remote_addr,
        remote_rkey,
        IbvWrOpcode::RdmaWrite,
    );


    println!("{} plugin_isend id {} con_id {} mr addr {}, remote addr {}, nreq {}", get_hostname(), state.id, sender.connection_id(), mr.addr(), remote_addr, state.nreqs);
    //thread::sleep(std::time::Duration::from_secs(5));
    if let Err(e) = sender.qp_list[0].ibv_post_send(send_wr.as_ptr()){
        log::error!("Error posting send: {:?}", e);
        return 1;
    }
    /*
    if let Err(e) = sender.qp_list[0].complete(1, IbvWcOpcode::RdmaWrite, SendRecv::Send){
        log::error!("Error completing: {:?}", e);
        return 1;
    }
    */
    /*
    let notify_send_wr = IbvSendWr::new(&sender.out_buffer_mr(), sender.in_remote_buffer_addr(), sender.in_remote_buffer_rkey(), IbvWrOpcode::Send);
    if let Err(e) = sender.qp_list[0].ibv_post_send(notify_send_wr.inner){
        log::error!("Error posting recv: {:?}", e);
        return 1;
    }
    */
    if state.nreqs == state.mrs{
        state.nreqs = 0;
    } else {
        state.nreqs += 1;
    }
    Box::into_raw(mr);
    Box::into_raw(sender_receiver);
    let request = Request{
        id: rand::random::<u32>(),
        size: _size as u64,
        done: true,
        send_receive: 0,
        address: 0,
        rkey: 0,
    };
    
    unsafe { * _request = Box::into_raw(Box::new(request)) as *mut c_void };
    0
}
extern "C" fn plugin_irecv(recv_comm: *mut c_void, n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _tags: *mut c_int, mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    let mut sender_receiver = unsafe { Box::from_raw(recv_comm as *mut SenderReceiver) };
    let (receiver, state) = if let SenderReceiver::Receiver{ref mut receiver, ref mut state} = *sender_receiver{
        (receiver, state)
    } else {
        log::error!("Error accepting: {:?}", "Not a receiver");
        return 1;
    };
    //println!("{} plugin_irecv state {:#?}, {}", get_hostname(), state, receiver.mrs());
    //thread::sleep(std::time::Duration::from_millis(1000));

    /*
    if state.nreqs == state.mrs{
        match receiver.qp_list[0].complete(0, IbvWcOpcode::RecvRdmaWithImm, SendRecv::Recv){
            Ok(completed) => {
                state.completed += completed as u32;
            },
            Err(e) => {
                log::error!("Error completing: {:?}", e);
                return 1;
            }
        }
        if state.completed == state.posted{
            *state = State::default();
            let request = Request{
                id: rand::random::<u32>(),
                size: 0,
                done: true,
                send_receive: 0,
                address: 0,
                rkey: 0,
            };
            unsafe { * _request = Box::into_raw(Box::new(request)) as *mut c_void };
        }
    } else {
        let mut size = 0;
        let mut address_list = Vec::new();
        for i in 0..n {
            let s = unsafe { * _sizes.offset(i as isize) };
            size += s;
            let mhandle = unsafe { *mhandles.offset(i as isize) };
            let mr = unsafe { Box::from_raw(mhandle as *mut IbvMr) };
            let out_nccl_metadata = receiver.out_buffer_ptr() as *mut NcclMetadata;
            let out_nccl_metadata: &mut NcclMetadata = unsafe { &mut *out_nccl_metadata };
            let request = Request{
                id: rand::random::<u32>(),
                size: mr.length() as u64,
                done: true,
                send_receive: 1,
                address: mr.addr(),
                rkey: mr.rkey(),
            };
            out_nccl_metadata.address = mr.addr();
            out_nccl_metadata.rkey = mr.rkey();
            out_nccl_metadata.length = mr.length() as u64;
            out_nccl_metadata.nreq = receiver.nreqs();
            out_nccl_metadata.requests[receiver.nreqs() as usize] = request;
            address_list.push(mr.addr());

            //let notify_wr = IbvRecvWr::new(&receiver.in_buffer_mr());
            //if let Err(e) = receiver.qp_list[0].ibv_post_recv(notify_wr){
            //    log::error!("Error posting recv: {:?}", e);
            //    return 1;
            //}

            Box::into_raw(mr);
        }

        if {state.nreqs += 1; state.nreqs} == state.mrs {
            println!("{} plugin_irecv  state {:#?}", get_hostname(), state);
            thread::sleep(std::time::Duration::from_secs(5));
        };

        let request = Request{
            id: rand::random::<u32>(),
            size: size as u64,
            done: if {state.nreqs += 1; state.nreqs} == state.mrs {true} else {false},
            send_receive: 0,
            address: 0,
            rkey: 0,
        };
    
        unsafe { * _request = Box::into_raw(Box::new(request)) as *mut c_void };
        state.nreqs += 1;
        state.posted += 1;
    }
    */

    let mut size = 0;
    let mut address_list = Vec::new();
    let mut _last_req = false;
    for i in 0..n {
        let idx = i as isize;
        let s = unsafe { * _sizes.offset(idx) };
        size += s;
        let mhandle = unsafe { *mhandles.offset(idx) };
        let mr = unsafe { Box::from_raw(mhandle as *mut IbvMr) };
        let out_nccl_metadata = receiver.out_buffer_ptr() as *mut NcclMetadata;
        let out_nccl_metadata: &mut NcclMetadata = unsafe { &mut *out_nccl_metadata };
        let request = Request{
            id: rand::random::<u32>(),
            size: mr.length() as u64,
            done: true,
            send_receive: 1,
            address: mr.addr(),
            rkey: mr.rkey(),
        };
        println!("{} plugin_irecv id {} con_id {} address {} req {}", get_hostname(), state.id, receiver.connection_id(), mr.addr(), state.nreqs);
        out_nccl_metadata.address = mr.addr();
        out_nccl_metadata.rkey = mr.rkey();
        out_nccl_metadata.length = mr.length() as u64;
        out_nccl_metadata.nreq = state.nreqs as u64;
        out_nccl_metadata.requests[state.nreqs as usize] = request;
        address_list.push(mr.addr());

        //let notify_wr = IbvRecvWr::new(&receiver.in_buffer_mr());
        //if let Err(e) = receiver.qp_list[0].ibv_post_recv(notify_wr){
        //    log::error!("Error posting recv: {:?}", e);
        //    return 1;
        //}
        let last_req = if state.nreqs == state.mrs{
            //println!("{} plugin_irecv state {:#?}", get_hostname(), state);
            //println!("{} plugin_irecv Out metadata {:#?}", get_hostname(), out_nccl_metadata);
            //thread::sleep(std::time::Duration::from_secs(5));
            true
        } else {
            false
        };
        _last_req = last_req;

        Box::into_raw(mr);
    }






    let request = Request{
        id: rand::random::<u32>(),
        size: size as u64,
        done: true,
        send_receive: 0,
        address: 0,
        rkey: 0,
    };

    unsafe { * _request = Box::into_raw(Box::new(request)) as *mut c_void };
    if state.nreqs == state.mrs{
        state.nreqs = 0;
    } else {
        state.nreqs += 1;
    }
    state.posted += 1;

    Box::into_raw(sender_receiver);
    0
}
extern "C" fn plugin_iflush(_recv_comm: *mut c_void, _n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    0
}
extern "C" fn plugin_test(_request: *mut c_void, _done: *mut c_int, _size: *mut c_int) -> ncclResult_t {
    let req = unsafe { Box::from_raw(_request as *mut Request) };
    if req.done{
        unsafe { _done.write(1) } ;
    }
    0
}
extern "C" fn plugin_close_send(_send_comm: *mut c_void) -> ncclResult_t { 
    0
}
extern "C" fn plugin_close_recv(_recv_comm: *mut c_void) -> ncclResult_t { 
    0
}
extern "C" fn plugin_close_listen(_listen_comm: *mut c_void) -> ncclResult_t { 
    0
}
extern "C" fn plugin_irecv_consumed(_recv_comm: *mut c_void, _n: c_int, _request: *mut c_void) -> ncclResult_t { 
    0
}
extern "C" fn plugin_get_device_mr(_comm: *mut c_void, _mhandle: *mut c_void, _dptr_mhandle: *mut *mut c_void) -> ncclResult_t {
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


fn get_hostname() -> String {
    let hostname = hostname::get().unwrap();
    hostname.into_string().unwrap()
}

pub fn get_device_properties(dev_name: String) -> anyhow::Result<ncclNetProperties_v8_t, CustomError> {
    let device_list: *mut *mut ibv_device = unsafe { __ibv_get_device_list(null_mut()) };
    let mut num_devices = 0;
    while !unsafe { *device_list.offset(num_devices) }.is_null() {
        num_devices += 1;
    }
    if num_devices == 0 {
        return Err(CustomError::new("ibv_get_device_list".to_string(), -1).into());
    }
    for i in 0..num_devices {
        let device: *mut ibv_device = unsafe { *device_list.offset(i as isize) };
        let device_ctx = unsafe { ibv_open_device(device) };
        if device_ctx == null_mut() {
            return Err(CustomError::new("Failed to open device".to_string(), -1));
        }
        let device_name = unsafe { CStr::from_ptr((*device).name.as_ptr()) };
        let dev_name_string = device_name.to_str().unwrap();
        if dev_name_string != dev_name {
            continue;
        }
        let mut device_attr = unsafe { std::mem::zeroed::<ibv_device_attr>() };
        let ret = unsafe { ibv_query_device(device_ctx, &mut device_attr) };
        if ret != 0 {
            return Err(CustomError::new("Failed to query device".to_string(), ret));
        }
        let pci_path = format!("/sys/class/infiniband/{}/device", device_name.to_str().unwrap());
        // The speed field indicates the speed of the network port in Mbps (10^6 bits per second)
        // 400 gbps
        let speed = 400000;
        let props = ncclNetProperties_v8_t{
            name: device_name.as_ptr() as *mut i8,
            pciPath: pci_path.as_ptr() as *mut i8,
            guid: device_attr.node_guid,
            ptrSupport: (NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF) as i32,
            speed,
            port: 1,
            maxComms: 1,
            regIsGlobal: 0,
            latency: 0.0,
            maxRecvs: 1,
            netDeviceType: 0,
            netDeviceVersion: 0,
        };
        return Ok(props);
    
    }
    Err(CustomError::new("Device not found".to_string(), -1))

}

#[derive(Debug)]
pub struct CustomError{
    message: String,
    code: i32
}

impl CustomError{
    pub fn new(message: String, code: i32) -> CustomError{
        CustomError{message, code}
    }
    pub fn code(&self) -> i32{
        self.code
    }
}

impl Display for CustomError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        write!(f, "Error: {} with code: {}", self.message, self.code)
    }
}

#[derive(Debug, Clone, Default, Copy)]
pub struct Request{
    id: u32,
    size: u64,
    done: bool,
    send_receive: u8,
    address: u64,
    rkey: u32,
}

impl Request{
    const LEN: usize = std::mem::size_of::<Self>();
}

#[derive(Clone, Debug, Default)]
pub struct NcclMetadata{
    pub address: u64,
    pub rkey: u32,
    pub lkey: u32,
    pub length: u64,
    pub nreq: u64,
    pub requests: [Request; 8],
}

impl From<IbvMr> for NcclMetadata{
    fn from(mr: IbvMr) -> Self {
        let address = mr.addr();
        let nccl_metadata = address as *const NcclMetadata;
        let nccl_metadata: &NcclMetadata = unsafe { &*nccl_metadata };
        nccl_metadata.clone()
    }
}

impl From<Box<IbvMr>> for NcclMetadata{
    fn from(mr: Box<IbvMr>) -> Self {
        let address = mr.addr();
        let nccl_metadata = address as *const NcclMetadata;
        let nccl_metadata: &NcclMetadata = unsafe { &*nccl_metadata };
        nccl_metadata.clone()
    }
}

impl NcclMetadata{
    const LEN: usize = std::mem::size_of::<Self>();
}

impl ControlBufferTrait for NcclMetadata{
    fn length(&self) -> usize {
        std::mem::size_of::<Self>()
    }
    fn new() -> Pin<Box<dyn ControlBufferTrait>> where Self: Sized {
        let nccl_metadata = NcclMetadata{
            address: rand::random::<u64>(),
            rkey: 0,
            lkey: 0,
            length: 0,
            nreq: 0,
            requests: [Request::default(); 8],
        };
        Box::pin(nccl_metadata)
    }
    fn size() -> usize where Self: Sized {
        std::mem::size_of::<Self>()
    }
    fn address(&self) -> u64 {
        self.address
    }
}