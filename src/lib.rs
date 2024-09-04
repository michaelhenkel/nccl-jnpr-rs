use std::ffi::{c_void, CStr};
use std::fmt::Display;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::os::raw::c_char;
use std::pin::Pin;
use std::ptr::null_mut;
use std::{thread, time};
use std::fmt::Debug;

use bincode::de::read;
use ibverbs_rs::receiver::Receiver;
use ibverbs_rs::sender::Sender;
use ibverbs_rs::{print_wr_ids, ControlBufferTrait, Hints, IbvAccessFlags, IbvMr, IbvQp, IbvRecvWr, IbvSendWr, IbvWcOpcode, IbvWrOpcode, LookUpBy, QpMode, SendRecv, SLOT_COUNT};
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

#[derive(Debug)]
struct State{
    id: u32,
    mrs: u32,
    nreqs: u32,
    completed: u32,
    request_id: u32,
    sub_request_id: u32,
    posted: u32,
    requests: [Box<Request>; 128],
    req_idx: usize,
    total_size: u64,
    transferred: u64,
    remaining: u64,
    slots: u32,
    slots_retrieved: bool,
    current_slot: u32,
    remaining_slots: u32,
}

impl Default for State{
    fn default() -> Self {
        let mut requests_vec: Vec<Box<Request>> = Vec::with_capacity(128);
        for _ in 0..128 {
            requests_vec.push(Box::new(Request::default()));
        }
        let requests: [Box<Request>; 128] = requests_vec.try_into().unwrap();
        for i in 0..128{
            let mut request = Request::default();
            request.id = i as u32;
        }
        State{
            id: 0,
            mrs: 0,
            nreqs: 0,
            request_id: 0,
            sub_request_id: 0,
            completed: 0,
            posted: 0,
            requests,
            req_idx: 0,
            total_size: 0,
            transferred: 0,
            remaining: 0,
            slots: 0,
            slots_retrieved: false,
            current_slot: 0,
            remaining_slots: 0,
        }
    }
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
    let mut receiver = match Receiver::new::<NcclMetadataList>(lookup_by, port, qp_mode){
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

    let mut sender = match Sender::new::<NcclMetadataList>(
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
    let in_buffer_ptr = sender.in_buffer_ptr();
    let out_buffer_ptr = sender.out_buffer_ptr();
    let in_buffer_mr = sender.in_buffer_mr();
    let out_buffer_mr = sender.out_buffer_mr();

    println!("{} isend con_id {} in_buffer_ptr {} out_buffer_ptr {} in_buffer_mr {} out_buffer_mr {}",
            get_hostname(),
            sender.connection_id(),
            in_buffer_ptr as u64,
            out_buffer_ptr as u64,
            in_buffer_mr.addr(),
            out_buffer_mr.addr()
    );
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
    let (pd, _sender_recv, _state) = match *sender_receiver{
        SenderReceiver::Sender{ref mut sender, ref mut state} => {
            state.mrs += 1;
            sender.incr_mrs();
            (sender.pd.clone(), "sender".to_string(), state)

        },
        SenderReceiver::Receiver{ref mut receiver, ref mut state} => {
            state.mrs += 1;
            receiver.incr_mrs();
            (receiver.pd.clone(), "recv".to_string(), state)
        }
    };
    let mr = IbvMr::new(pd, data, size, access_flags);
    println!("{} plugin_reg_mr {}, state_id {}, mr addr {}, size {},", get_hostname(),_sender_recv, _state.id, mr.addr(), size);
    let mr_handle = Box::into_raw(Box::new(mr));
    unsafe { *mhandle = mr_handle as *mut c_void; }
    Box::into_raw(sender_receiver); 
    0
}
extern "C" fn plugin_reg_mr_dma_buf(_coll_comm: *mut c_void, _data: *mut c_void, _size: size_t, _type_: c_int, _offset: u64, _fd: c_int, _mhandle: *mut *mut c_void) -> ncclResult_t {
    println!("{} plugin_reg_mr_dma_buf", get_hostname());
    0
}
extern "C" fn plugin_dereg_mr(_coll_comm: *mut c_void, _mhandle: *mut c_void) -> ncclResult_t { 
    0
}

extern "C" fn plugin_isend(send_comm: *mut c_void, _data: *mut c_void, _size: c_int, _tag: c_int, _mhandle: *mut c_void, mut _request: *mut *mut c_void) -> ncclResult_t { 
    let mut sender_receiver = unsafe { Box::from_raw(send_comm as *mut SenderReceiver) };
    let (sender, state) = if let SenderReceiver::Sender{ref mut sender, ref mut state} = *sender_receiver{
        (sender, state)
    } else {
        println!("{} plugin_isend error {:?}", get_hostname(), "Not a sender");
        log::error!("Error accepting: {:?}", "Not a sender");
        return 1;
    };
    let mr = unsafe { Box::from_raw(_mhandle as *mut IbvMr) };
    let slots = mr.length() / _size as usize;
    let remainder = mr.length() % _size as usize;
    let slots: usize = if remainder > 0 { slots + 1 } else { slots };
    state.slots = slots as u32;


    let request_idx = state.req_idx;
    let mut request = state.requests[request_idx].clone();
    


    let magix: u32 = rand::random();

    let current_slot = state.current_slot;

    let slot_addr = sender.in_buffer_mr().addr() + current_slot as u64 * NcclMetadata::LEN as u64;
    let nccl_metadata = slot_addr as *mut NcclMetadata;
    let nccl_metadata: &mut NcclMetadata = unsafe { &mut *nccl_metadata };
    while nccl_metadata.address == 0 || nccl_metadata.rkey == 0 || nccl_metadata.length == 0 {
        unsafe { *_request = null_mut() };
        Box::into_raw(mr);
        Box::into_raw(sender_receiver);
        return 0;
        //thread::sleep(std::time::Duration::from_secs(1));
    }
    println!("{} isend con_id {} slot {} magix {} addr {} rkey {} lkey {} length {} nreq {} sub_req_id {} req_id {}",
        get_hostname(),
        sender.connection_id(),
        current_slot,
        magix,
        nccl_metadata.address,
        nccl_metadata.rkey,
        nccl_metadata.lkey,
        nccl_metadata.length,
        nccl_metadata.nreq,
        nccl_metadata.request.id,
        nccl_metadata.request.request_id
    );  
    


    let remote_addr = nccl_metadata.request.address;
    let remote_rkey = nccl_metadata.request.rkey;
    request.id = magix;
    request.request_id = nccl_metadata.request.request_id;
    request.size = _size as u64;
    request.sizes[0] = _size;
    request.n = 1;
    request.done = true;
    request.slot = current_slot;
    request.total_slots = state.slots;
    request.sender_receiver = RequestSenderReceiver::Sender;
    request.address = remote_addr;
    request.connection_id = sender.connection_id();
    request.state_id = state.id;
    request.nreq = state.nreqs;
    request.qp = Some(sender.qp_list[0].clone());

    state.total_size = mr.length() as u64;

    let signaled = if state.slots == state.current_slot + 1 { 
        true 
    } else { 
        false
    };

    request.complete = signaled;

    let send_wr = ibverbs_rs::IbvSendWr::new(
        mr.addr(),
        mr.lkey(),
        remote_addr,
        remote_rkey,
        _size as u64,
        state.transferred,
        IbvWrOpcode::RdmaWrite,
        signaled,
    );

    /*
    println!("{} isend con_id {} l_addr {} offset {} r_addr {} tot {} tx {} rem {} req {} tx_size {} mr_size {}",
        get_hostname(),
        sender.connection_id(),
        mr.addr(),
        state.transferred,
        remote_addr,
        state.total_size,
        state.transferred,
        state.remaining,
        state.nreqs,
        _size,
        mr.length()
    );
    */
    
    
    //thread::sleep(std::time::Duration::from_millis(100));


    if let Err(e) = sender.qp_list[0].ibv_post_send(send_wr.as_ptr()){
        println!("{} plugin_isend post error {:#?}", get_hostname(), e);
        log::error!("Error posting send: {:?}", e);
        return 1;
    }
    

    nccl_metadata.address = 0;
    nccl_metadata.rkey = 0;
    nccl_metadata.length = 0;

    state.current_slot += 1;
    state.remaining_slots = state.slots - state.current_slot;
    state.transferred += _size as u64;
    state.remaining = state.total_size - state.transferred;
    
    if state.remaining == 0{
        for i in 0..state.nreqs{
            let mut request = Request::default();
            request.id = i as u32;
            state.requests[i as usize] = Box::new(request);
        }
        state.total_size = 0;
        state.transferred = 0;
        state.remaining = 0;
        state.nreqs = 0;
        state.req_idx = 0;
        state.slots = 0;
        state.current_slot = 0;
        state.remaining_slots = 0;
        state.slots_retrieved = false;
    } else {
        state.nreqs += 1;
        state.req_idx += 1;
    }

    if request.qp.is_none(){
        panic!("{} isend id {} con_id {} request {:?} qp is none", get_hostname(), state.id, sender.connection_id(), request);
    }
    
    Box::into_raw(mr);
    Box::into_raw(sender_receiver);
    unsafe { * _request = Box::into_raw(request) as *mut c_void };
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
    let magix: u32 = rand::random();

    /*
    let recv_wr = IbvRecvWr::new(None, Some(magix as u64));
    if let Err(e) = receiver.qp_list[0].ibv_post_recv(recv_wr){
        log::error!("Error posting receive: {:?}", e);
        return 1;
    }
    */
    if state.request_id == 0{
        state.request_id = rand::random::<u32>();
    }


    let request_idx = state.req_idx;
    let mut request = state.requests[request_idx].clone();
    let mut address_list = Vec::new();
    for i in 0..n {
        let idx = i as isize;
        let s = unsafe { * _sizes.offset(idx) };
        let mhandle = unsafe { *mhandles.offset(idx) };
        let mr = unsafe { Box::from_raw(mhandle as *mut IbvMr) };

        let slots = mr.length() / s as usize;
        let remainder = mr.length() % s as usize;
        let slots: usize = if remainder > 0 { slots + 1 } else { slots };
        state.slots = slots as u32;

        let out_nccl_metadata_list = receiver.out_buffer_ptr() as *mut NcclMetadataList;
        let out_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *out_nccl_metadata_list };

        state.total_size = mr.length() as u64;



        let offset = state.transferred;
        request.id = magix;
        request.size = s as u64;
        request.sizes[i as usize] = s;
        request.n = n as u32;
        request.done = true;
        request.sender_receiver = RequestSenderReceiver::Receiver;
        request.address = mr.addr();
        request.offset = offset;
        request.offset_address = mr.addr() + offset;
        request.rkey = mr.rkey();
        request.state_id = state.id;
        request.connection_id = receiver.connection_id();
        request.nreq = state.nreqs;
        request.qp = Some(receiver.qp_list[0].clone());
        request.magix = magix;
        request.slot = state.current_slot;
        request.total_slots = state.slots;
        request.md_local_address = receiver.out_buffer_mr().addr();
        request.md_remote_address = receiver.in_remote_buffer_addr();
        request.md_offset = NcclMetadata::LEN as u64 * state.current_slot as u64;
        request.md_lkey = receiver.out_buffer_mr().lkey();
        request.md_rkey = receiver.in_remote_buffer_rkey();
        request.md_length = NcclMetadata::LEN as u64;
        request.md_local_offset_addr = receiver.out_buffer_mr().addr() + request.md_offset;
        request.md_remote_offset_addr = receiver.in_remote_buffer_addr() + request.md_offset;
        request.request_id = state.request_id;
        //let mr_size = mr.length();
        //println!("{} plugin_irecv id {} con_id {} transferred {} req {} mr_size {} data_size {}", get_hostname(), state.id, receiver.connection_id(), offset, state.nreqs, mr_size, s);
        
        //println!("{} plugin_irecv id {} nccl_metadata_list_len {}", get_hostname(), state.id, out_nccl_metadata_list.len());
        let out_nccl_metadata = out_nccl_metadata_list.0.get_mut(state.current_slot as usize).unwrap();
        out_nccl_metadata.address = mr.addr() + offset;
        out_nccl_metadata.rkey = mr.rkey();
        out_nccl_metadata.length = s as u64;
        out_nccl_metadata.nreq = state.nreqs as u64;
        out_nccl_metadata.request = *request.clone();

        let signaled = if state.slots == state.current_slot + 1 { true } else { false };

        request.complete = signaled;

        let send_wr = ibverbs_rs::IbvSendWr::new(
            receiver.out_buffer_mr().addr(),
            receiver.out_buffer_mr().lkey(),
            receiver.in_remote_buffer_addr(),
            receiver.in_remote_buffer_rkey(),
            NcclMetadata::LEN as u64,
            NcclMetadata::LEN as u64 * state.current_slot as u64,
            IbvWrOpcode::RdmaWrite,
            signaled,
        );

        println!("{} irecv con_id {} slot {} magix {} r_addr {} l_addr {} offset {} size {} mr_addr {} mr_length {} mr_rkey {}",
            get_hostname(),
            receiver.connection_id(),
            state.current_slot,
            magix,
            receiver.in_remote_buffer_addr(),
            receiver.out_buffer_mr().addr(),
            NcclMetadata::LEN as u64 * state.current_slot as u64,
            NcclMetadata::LEN as u64,
            mr.addr(),
            mr.length(),
            mr.rkey()
        );

        if let Err(e) = receiver.qp_list[0].ibv_post_send(send_wr.as_ptr()){
            println!("{} plugin_irecv post error {:#?}", get_hostname(), e);
            log::error!("Error posting send: {:?}", e);
            return 1;
        }

        address_list.push(offset);
        state.transferred += s as u64;
        state.current_slot += 1;
        state.remaining = state.total_size - state.transferred;
        state.remaining_slots = state.slots - state.current_slot;


        Box::into_raw(mr);
    }

    if request.qp.is_none(){
        panic!("{} irecv id {} con_id {} request {:?} qp is none", get_hostname(), state.id, receiver.connection_id(), request);
    }

    let cloned_request = request.clone();
    state.requests[request_idx] = request;
    if state.remaining == 0{
        for i in 0..state.nreqs{
            let mut request = Request::default();
            request.id = i as u32;
            state.requests[i as usize] = Box::new(request);
        }
        state.total_size = 0;
        state.transferred = 0;
        state.remaining = 0;
        state.nreqs = 0;
        state.req_idx = 0;
        state.slots = 0;
        state.current_slot = 0;
        state.remaining_slots = 0;
        state.request_id = 0;
        
    } else {
        state.nreqs += 1;
        state.req_idx += 1;
        
    }
    state.posted += 1;

    unsafe { * _request = Box::into_raw(cloned_request) as *mut c_void };
    
    Box::into_raw(sender_receiver);
    0
}
extern "C" fn plugin_iflush(_recv_comm: *mut c_void, _n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    let req = unsafe { Box::from_raw(_request as *mut Request) };
    println!("{} plugin_iflush id {} con_id {} address {} req {}", get_hostname(), req.id, 0, req.address, 0);
    //thread::sleep(std::time::Duration::from_secs(5));
    0
}
extern "C" fn plugin_test(mut _request: *mut c_void, _done: *mut c_int, _size: *mut c_int) -> ncclResult_t {
    let req = unsafe { Box::from_raw(_request as *mut Request) };
    let qp = req.qp.clone().unwrap();
    let opcode = req.opcode.clone();
    let send_recv = req.send_recv.clone();
    let sizes = req.sizes;

    match req.sender_receiver{
        RequestSenderReceiver::Receiver => {
            println!("{} test receiver {:?} opcode {:?}", get_hostname(),req, opcode);
            thread::sleep(std::time::Duration::from_secs(2));
            if req.complete {
                println!("{} test receiver waiting for completion", get_hostname());
                match qp.complete(1, opcode, send_recv, None){
                    Ok(completed) => {
                        println!("{} test receiver got {} completions", get_hostname(), completed);
                        if completed == 0 {
                            unsafe { _done.write(0) };
                            return 0;
                        } 
                    },
                    Err(e) => {
                        println!("{} plugin_test complete error {:#?}", get_hostname(), e);
                        log::error!("Error completing: {:?}", e);
                        return 1;
                    }
                }

            } else {
                //println!("{} test s_r con_id {} magix {} sender_receiver {:?} waiting for completion", get_hostname(),req.connection_id, req.magix, req.sender_receiver);
            }
        },
        RequestSenderReceiver::Sender => {
            println!("{} test sender {:?} opcode {:?}", get_hostname(),req, opcode);
            thread::sleep(std::time::Duration::from_secs(2));
            if req.complete{
                println!("{} test sender waiting for completion", get_hostname());
                match qp.complete(1, opcode, send_recv, None){
                    Ok(completed) => {
                        println!("{} test sender got {} completions", get_hostname(), completed);
                        if completed == 0 {
                            unsafe { _done.write(0) };
                            return 0;
                        }
                    },
                    Err(e) => {
                        println!("{} plugin_test complete error {:#?}", get_hostname(), e);
                        log::error!("Error completing: {:?}", e);
                        return 1;
                    }
                }

            } else {
                //println!("{} test s_r con_id {} magix {} sender_receiver {:?} waiting for completion", get_hostname(),req.connection_id, req.magix, req.sender_receiver);
            }
        }
    };



   
    unsafe { _done.write(1) };
    unsafe { std::ptr::copy_nonoverlapping(sizes.as_ptr(), _size, req.n as usize) };
    _request = null_mut();
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
            maxComms: 1024 * 1024,
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

#[derive(Clone, Debug)]
pub struct NcclMetadataList(pub [NcclMetadata; SLOT_COUNT]);

impl NcclMetadataList{
    const LEN: usize = std::mem::size_of::<Self>();
}

impl From<IbvMr> for NcclMetadataList{
    fn from(mr: IbvMr) -> Self {
        let address = mr.addr();
        let size = mr.length();
        let mut nccl_metadata_list = Vec::new();
        let nccl_metadata_size = NcclMetadata::LEN;
        let slots = size / nccl_metadata_size;
        let remainder = size % nccl_metadata_size;
        for i in 0..slots{
            let nccl_metadata = address + i as u64 * nccl_metadata_size as u64;
            let nccl_metadata = nccl_metadata as *const NcclMetadata;
            let nccl_metadata: &NcclMetadata = unsafe { &*nccl_metadata };
            nccl_metadata_list.push(nccl_metadata.clone());
        }
        if remainder > 0{
            let nccl_metadata = address + slots as u64 * nccl_metadata_size as u64;
            let nccl_metadata = nccl_metadata as *const NcclMetadata;
            let nccl_metadata: &NcclMetadata = unsafe { &*nccl_metadata };
            nccl_metadata_list.push(nccl_metadata.clone());
        }
        let nccl_metadata_list = nccl_metadata_list.try_into().unwrap();
        NcclMetadataList(nccl_metadata_list)
    }
}

impl From<Box<IbvMr>> for NcclMetadataList{
    fn from(mr: Box<IbvMr>) -> Self {
        let address = mr.addr();
        let size = mr.length();
        let mut nccl_metadata_list = Vec::new();
        let nccl_metadata_size = NcclMetadata::LEN;
        let slots = size / nccl_metadata_size;
        let remainder = size % nccl_metadata_size;
        for i in 0..slots{
            let nccl_metadata = address + i as u64 * nccl_metadata_size as u64;
            let nccl_metadata = nccl_metadata as *const NcclMetadata;
            let nccl_metadata: &NcclMetadata = unsafe { &*nccl_metadata };
            nccl_metadata_list.push(nccl_metadata.clone());
        }
        if remainder > 0{
            let nccl_metadata = address + slots as u64 * nccl_metadata_size as u64;
            let nccl_metadata = nccl_metadata as *const NcclMetadata;
            let nccl_metadata: &NcclMetadata = unsafe { &*nccl_metadata };
            nccl_metadata_list.push(nccl_metadata.clone());
        }
        let nccl_metadata_list = nccl_metadata_list.try_into().unwrap();
        NcclMetadataList(nccl_metadata_list)
    }
}

impl ControlBufferTrait for NcclMetadataList{
    fn length(&self) -> usize {
        std::mem::size_of::<Self>()
    }
    fn new() -> Pin<Box<dyn ControlBufferTrait>> where Self: Sized {
        let mut nccl_metadata_list = Vec::with_capacity(SLOT_COUNT);
        for _ in 0..SLOT_COUNT{
            nccl_metadata_list.push(NcclMetadata::default());
        }
        let nccl_metadata_list: [NcclMetadata; SLOT_COUNT] = nccl_metadata_list.try_into().unwrap();
        let nccl_metadata_list = NcclMetadataList(nccl_metadata_list);
        Box::pin(nccl_metadata_list)
    }
    fn size() -> usize where Self: Sized {
        std::mem::size_of::<Self>()
    }
    fn address(&self) -> u64 {
        self.0.as_ptr() as u64
    }
    fn print(&self){
        for nccl_metadata in self.0.iter(){
            println!("{} address {} rkey {} lkey {} length {} nreq {} req_id {}",
                get_hostname(),
                nccl_metadata.address,
                nccl_metadata.rkey,
                nccl_metadata.lkey,
                nccl_metadata.length,
                nccl_metadata.nreq,
                nccl_metadata.request.id
            );
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NcclMetadata{
    pub address: u64,
    pub rkey: u32,
    pub lkey: u32,
    pub length: u64,
    pub nreq: u64,
    pub request: Request,
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

#[derive(Clone, Debug)]
enum RequestSenderReceiver{
    Sender,
    Receiver
}

#[derive(Clone)]
pub struct Request{
    id: u32,
    state_id: u32,
    n: u32,
    nreq: u32,
    slot: u32,
    total_slots: u32,
    connection_id: u32,
    size: u64,
    done: bool,
    sender_receiver: RequestSenderReceiver,
    address: u64,
    offset: u64,
    offset_address: u64,
    rkey: u32,
    sizes: [i32; 8],
    qp: Option<IbvQp>,
    opcode: IbvWcOpcode,
    send_recv: SendRecv,
    magix: u32,
    md_local_address: u64,
    md_remote_address: u64,
    md_offset: u64,
    md_local_offset_addr: u64,
    md_remote_offset_addr: u64,
    md_rkey: u32,
    md_lkey: u32,
    md_length: u64,
    complete: bool,
    request_id: u32,

}

impl Debug for Request{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {{ id: {}, state_id: {}, n: {}, nreq: {}, slot: {}, total_slots: {}, connection_id: {}, size: {}, done: {}, sender_receiver: {:?}, address: {}, offset: {}, offset_addr: {} rkey: {}, sizes: {:?} magix {} md_local_address {} md_remote_address {} md_offset {} md_local_offset_addr {} md_remote_offset_addr {} md_rkey {} md_lkey {} md_length {} complete {} request_id {}}}",
            self.id,
            self.state_id,
            self.n,
            self.nreq,
            self.slot,
            self.total_slots,
            self.connection_id,
            self.size,
            self.done,
            self.sender_receiver,
            self.address,
            self.offset,
            self.offset_address,
            self.rkey,
            self.sizes,
            self.magix,
            self.md_local_address,
            self.md_remote_address,
            self.md_offset,
            self.md_local_offset_addr,
            self.md_remote_offset_addr,
            self.md_rkey,
            self.md_lkey,
            self.md_length,
            self.complete,
            self.request_id,
        )
    }
}

impl Default for Request{
    fn default() -> Self {
        Request{
            id: 0,
            state_id: 0,
            n: 0,
            nreq: 0,
            slot: 0,
            total_slots: 0,
            connection_id: 0,
            size: 0,
            done: false,
            sender_receiver: RequestSenderReceiver::Sender,
            address: 0,
            offset: 0,
            offset_address: 0,
            rkey: 0,
            sizes: [0; 8],
            qp: None,
            opcode: IbvWcOpcode::RdmaWrite,
            send_recv: SendRecv::Send,
            magix: 0,
            md_local_address: 0,
            md_remote_address: 0,
            md_offset: 0,
            md_local_offset_addr: 0,
            md_remote_offset_addr: 0,
            md_rkey: 0,
            md_lkey: 0,
            md_length: 0,
            complete: false,
            request_id: 0,
        }
    }
}