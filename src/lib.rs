use std::{
    collections::HashMap, ffi::{c_void, CStr}, fmt::{Debug, Display}, net::{IpAddr, Ipv4Addr, Ipv6Addr}, os::{raw::c_char, unix::thread}, pin::Pin, ptr::null_mut, sync::{
        atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, Once
    }
};
use ibverbs_rs::{
    debug_wr, print_wr_ids, receiver::{Receiver, ReceiverInterface}, sender::{Sender, SenderInterface}, ControlBufferTrait, Hints, IbvAccessFlags, IbvMr, IbvQp, IbvRecvWr, IbvSendWrList, IbvWcOpcode, IbvWrOpcode, LookUpBy, QpMode, SendRecv, WrsDebug, SLOT_COUNT
};
use rdma_sys::*;
use env_logger::Env;
mod bindings;
use bindings::*;
use serde::{Serialize, Deserialize};
use log::info as rust_info;

static INIT: Once = Once::new();
pub fn initialize_logger() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(Env::default().default_filter_or("info"))
            .init();
    });
}

use libc::{c_int, size_t};

static mut LOG_FUNCTION: ncclDebugLogger_t = None;

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        rust_warn!($($arg)*);
        if let Some(log_function) = unsafe { LOG_FUNCTION } {
            let file = std::ffi::CString::new(file!()).unwrap();
            let line = line!() as c_int;
            let format = std::ffi::CString::new(format!($($arg)*)).unwrap();
            unsafe {
                log_function(ncclDebugLogLevel_NCCL_LOG_WARN, NCCL_ALL, file.as_ptr(), line, format.as_ptr());
            }
        }
    }};
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        rust_info!($($arg)*);
        if let Some(log_function) = unsafe { LOG_FUNCTION } {
            let func = std::ffi::CString::new(std::any::type_name::<fn()>()).unwrap();
            let line = line!() as c_int;
            let format = std::ffi::CString::new(format!($($arg)*)).unwrap();
            unsafe {
                log_function(ncclDebugLogLevel_NCCL_LOG_INFO, 0, func.as_ptr(), line, format.as_ptr());
            }
        }
    }};
}

#[macro_export]
macro_rules! abort {
    ($flags:expr, $($arg:tt)*) => {{
        rust_info!($($arg)*);
        if let Some(log_function) = unsafe { LOG_FUNCTION } {
            let func = std::ffi::CString::new(std::any::type_name::<fn()>()).unwrap();
            let line = line!() as c_int;
            let format = std::ffi::CString::new(format!($($arg)*)).unwrap();
            unsafe {
                log_function(ncclDebugLogLevel_NCCL_LOG_ABORT, $flags, func.as_ptr(), line, format.as_ptr());
            }
        }
    }};
}

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

#[derive(Clone)]
struct Request{
    request_type: RequestType,
    connection_id: u32,
    stage: RequestStage,
    idx: u32,
    sizes: Vec<u32>,
    id: u64,
    completed: bool,
    md_completed: bool,
    md_idx: u32,
    completions: u32,
}

impl Request{
    fn new() -> Self{
        Request{
            request_type: RequestType::Available,
            connection_id: 0,
            stage: RequestStage::SendMetadata,
            idx: 0,
            sizes: Vec::new(),
            id: 0,
            completed: false,
            md_completed: false,
            md_idx: 0,
            completions: 0,
        }
    }
    fn reset(&mut self){
        self.request_type = RequestType::Available;
        self.connection_id = 0;
        self.stage = RequestStage::SendMetadata;
        self.sizes = Vec::new();
        self.id = 0;
        self.completed = false;
        self.md_completed = false;
        self.md_idx = 0;
        self.completions = 0;
    }
}

impl Debug for Request{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {{ request_type: {:?}, connection_id: {}, stage: {:?}, idx: {}, sizes: {:?}, id: {}, completed: {}, md_completed: {}, completions: {} }}",
            self.request_type,
            self.connection_id,
            self.stage,
            self.idx,
            self.sizes,
            self.id,
            self.completed,
            self.md_completed,
            self.completions,
        )
    }
}

#[derive(Debug, Clone)]
enum RequestStage{
    SendMetadata,
    WaitForData,
    WaitForSendCompletion,
    Finished,
}

#[derive(Debug, Default, Clone)]
enum RequestType{
    SendData,
    RecvData,
    #[default]
    Available,
}

struct State{
    id: u32,
    connection_id: u32,
    recv_send: SendRecv,
    request_manager: Arc<RequestManager>,
    metadata_allocator: Arc<MetadataAllocator>,
    completion_tracker: Arc<CompletionTracker>,
    data_tracker: Arc<DataTracker>,
}

impl State{
    fn new(num_qps: usize) -> Self{
        State{
            id: 0,
            connection_id: 0,
            recv_send: SendRecv::Send,
            request_manager: RequestManager::new(),
            metadata_allocator: Arc::new(MetadataAllocator::new()),
            completion_tracker: Arc::new(CompletionTracker::new(num_qps)),
            data_tracker: Arc::new(DataTracker::new()),
        }
    }
}

impl Debug for State{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {{ id: {}, connection_id: {}, recv_send: {:?} }}",
            self.id,
            self.connection_id,
            self.recv_send,
        )
    }
}

extern "C" fn plugin_init(log_function: ncclDebugLogger_t) -> ncclResult_t {
    unsafe {
        LOG_FUNCTION = log_function;
    }
    initialize_logger();
    0
}
extern "C" fn plugin_devices(ndev: *mut c_int) -> ncclResult_t {
    unsafe { ndev.write(1) };
    
    0
}
extern "C" fn plugin_get_properties(_dev: c_int, props: *mut ncclNetProperties_v8_t) -> ncclResult_t {
    match get_device_properties("mlx5_3".to_string()){
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
    let lookup_by = LookUpBy::Name("mlx5_3".to_string());
    let qp_mode = QpMode::Multi;
    let num_qps = 4;
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
    let mut state = State::new(num_qps);
    state.id = rand::random::<u32>();
    state.connection_id = receiver.connection_id();
    state.recv_send = SendRecv::Recv;
    let listen_comm_handle = Box::into_raw(Box::new(SenderReceiver::Receiver { 
        receiver: ReceiverWrapper::new(Box::new(receiver)),
        state: Arc::new(state), 
    }));
    if !listen_comm.is_null() {
        unsafe {
            *listen_comm = listen_comm_handle as *mut c_void;
        }
    }
    0
}
extern "C" fn plugin_connect(_dev: c_int, handle: *mut c_void, send_comm: *mut *mut c_void, _send_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    let num_qps = 4;
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
    let lookup_by = LookUpBy::Name("mlx5_3".to_string());

    let mut sender = match Sender::new::<NcclMetadataList>(
        lookup_by,
        receiver_address,
        port,
        num_qps,
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
    let mut state = State::new(num_qps as usize);
    state.id = rand::random::<u32>();
    state.connection_id = sender.connection_id();
    state.recv_send = SendRecv::Send;
    let sender_handle = Box::into_raw(Box::new(SenderReceiver::Sender{
        sender: SenderWrapper::new(Box::new(sender)),
        state: Arc::new(state),
    }));

    unsafe {
        *send_comm = sender_handle as *mut c_void;
    }
    0
}
extern "C" fn plugin_accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void, _recv_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    let mut sender_receiver = unsafe { Box::from_raw(listen_comm as *mut SenderReceiver) };
    if let SenderReceiver::Receiver{ref mut receiver, state: _} = *sender_receiver{
        let receiver = receiver.receiver();
        let mut receiver = match receiver.lock(){
            Ok(receiver) => receiver,
            Err(poisoned) => {
                println!("poisoned");
                let receiver = poisoned.into_inner();
                receiver
            }
        };
        
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
    let (pd, _sender_recv, state) = match *sender_receiver{
        SenderReceiver::Sender{ref mut sender, ref mut state} => {
            let sender = sender.sender();
            let sender = match sender.lock(){
                Ok(sender) => sender,
                Err(poisoned) => {
                    println!("poisoned");
                    let sender = poisoned.into_inner();
                    sender
                }
            };
            (sender.pd(), "sender".to_string(), state)

        },
        SenderReceiver::Receiver{ref mut receiver, ref mut state} => {

            let receiver = receiver.receiver();
            let receiver = match receiver.lock(){
                Ok(receiver) => receiver,
                Err(poisoned) => {
                    println!("poisoned");
                    let receiver = poisoned.into_inner();
                    receiver
                }
            };

            (receiver.pd(), "recv".to_string(), state)
        }
    };

    

    let mr = IbvMr::new(pd, data, size, access_flags);
    //println!("{} plugin_reg_mr {}, state_id {}, mr addr {}, size {},", get_hostname(),_sender_recv, state.id, mr.addr(), size);
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
    let sender = sender.sender();
    let sender = match sender.lock(){
        Ok(sender) => sender,
        Err(poisoned) => {
            println!("poisoned");
            let sender = poisoned.into_inner();
            sender
        }
    };

    let request_manager = state.request_manager.clone();
    let (data_request_idx, _, request) = request_manager.create_request();
    let data_tracker = state.data_tracker.clone();
    data_tracker.increment_actual_send(_size as u64);
    let completion_tracker = state.completion_tracker.clone();
    let mut request_lock = request.lock().unwrap();
    request_lock.idx = data_request_idx as u32;
    request_lock.request_type = RequestType::SendData;
    request_lock.connection_id = state.connection_id;
    request_lock.stage = RequestStage::WaitForSendCompletion;
    request_lock.sizes = vec![_size as u32];

    let metadata_allocator = state.metadata_allocator.clone();
    let start_position = metadata_allocator.allocate(1);
    let in_nccl_metadata_list = sender.in_buffer_ptr() as *mut NcclMetadataList;
    let in_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *in_nccl_metadata_list };
    let in_nccl_metadata: &mut NcclMetadata = in_nccl_metadata_list.0.get_mut(start_position as usize).unwrap();

    while in_nccl_metadata.address == 0 || in_nccl_metadata.rkey == 0 {
        std::thread::sleep(std::time::Duration::from_micros(1));
        //info!("{} isend waiting for metadata", get_hostname());
    }


    let req_id = in_nccl_metadata.context;
    request_lock.id = req_id;
    let mhandle_mr = unsafe { Box::from_raw(_mhandle as *mut IbvMr) };

    let send_wr_list = ibverbs_rs::IbvSendWrList::new(
        _data as u64,
        mhandle_mr.lkey(),
        in_nccl_metadata.address,
        in_nccl_metadata.rkey,
        _size as u64,
        0,
        IbvWrOpcode::RdmaWrite,
        true,
        Some(data_request_idx),
        true,
        sender.num_qps() as u64,
    );

    for (qp_idx, qp) in sender.qps().iter().enumerate(){
        request_lock.completions += 1;
        //println!("{} isend wr_id {} qpn {} {:?}", get_hostname(), data_request_idx, qp.qp_num(), request_lock);
        completion_tracker.mark_uncomplete(data_request_idx, qp_idx);
        let send_wr = send_wr_list.get(qp_idx).unwrap();
        //print_wr_ids(send_wr.as_ptr());
        if let Err(e) = qp.ibv_post_send(send_wr.as_ptr()){
            println!("{} isend post error {:?}, {:?}", get_hostname(),e, request_lock);
            log::error!("Error posting send: {:?}", e);
            return 1;
        }
    }


    let test_request = TestRequest{
        qp_list: sender.qps(),
        request_manager: state.request_manager.clone(),
        completion_tracker: state.completion_tracker.clone(),
        wrs_debug: None,
        nccl_metadata: Some(in_nccl_metadata.clone()),
        request_idx: data_request_idx,
        data_tracker: state.data_tracker.clone(),
    };

    in_nccl_metadata.address = 0;
    in_nccl_metadata.rkey = 0;
    in_nccl_metadata.length = 0;
    in_nccl_metadata.nreqs = 0;
    in_nccl_metadata.idx = 0;
    in_nccl_metadata.context = 0;

    let test_request_box = Box::new(test_request);
    let test_request_ptr: *mut TestRequest = Box::into_raw(test_request_box);
    let test_request_ptr_as_c_void: *mut c_void = test_request_ptr as *mut c_void;
    unsafe { * _request = test_request_ptr_as_c_void };
    //std::thread::sleep(std::time::Duration::from_micros(100));
    Box::into_raw(mhandle_mr);
    Box::into_raw(sender_receiver);
    0
}

const MAX_REQUESTS: u32 = 255;

extern "C" fn plugin_irecv(recv_comm: *mut c_void, n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _tags: *mut c_int, mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    let mut sender_receiver = unsafe { Box::from_raw(recv_comm as *mut SenderReceiver) };
    let (receiver, state) = if let SenderReceiver::Receiver{ref mut receiver, ref mut state} = *sender_receiver{
        (receiver, state)
    } else {
        log::error!("Error accepting: {:?}", "Not a receiver");
        return 1;
    };

    let receiver = receiver.receiver();
    let receiver = match receiver.lock(){
        Ok(receiver) => receiver,
        Err(poisoned) => {
            println!("poisoned");
            let receiver = poisoned.into_inner();
            receiver
        }
    };

    let request_manager = state.request_manager.clone();
    let (data_request_idx, metadata_request_idx, request) = request_manager.create_request();

    let data_tracker = state.data_tracker.clone();

    let mut request_lock = request.lock().unwrap();
    request_lock.request_type = RequestType::RecvData;
    request_lock.connection_id = state.connection_id;
    request_lock.idx = data_request_idx as u32;
    request_lock.id = rand::random::<u32>() as u64;
    let metadata_allocator = state.metadata_allocator.clone();
    let start_position = metadata_allocator.allocate(n as usize);
    let completion_tracker = state.completion_tracker.clone();
    

    for i in 0..receiver.num_qps(){
        let recv_wr = IbvRecvWr::new(None, Some(data_request_idx));
        if let Err(e) = receiver.get_qp(i).ibv_post_recv(recv_wr){
            println!("Error posting receive: {:?}", e);
            return 1;
        }
        request_lock.completions += 1;
        completion_tracker.mark_uncomplete(data_request_idx, i);
    }



    let out_nccl_metadata_list = receiver.out_buffer_ptr() as *mut NcclMetadataList;
    let out_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *out_nccl_metadata_list };
    for idx in 0..n {
        let position = start_position + idx as usize;
        let size = unsafe { * _sizes.offset(idx as isize) };
        //data_tracker.increment_expected_recv(size as u64);
        let mhandle = unsafe { *mhandles.offset(idx as isize) };
        let mhandle_mr = unsafe { Box::from_raw(mhandle as *mut IbvMr) };
        let data = unsafe { * _data.offset(idx as isize) };
        let out_nccl_metadata = out_nccl_metadata_list.0.get_mut(position as usize).unwrap();
        out_nccl_metadata.address = data as u64;
        out_nccl_metadata.rkey = mhandle_mr.rkey();
        out_nccl_metadata.length = size as u64;
        out_nccl_metadata.nreqs = n as u64;
        out_nccl_metadata.idx = request_lock.idx as u64;
        out_nccl_metadata.context = request_lock.id;
        out_nccl_metadata.md_idx = position as u64;
        Box::into_raw(mhandle_mr);
    }

    //info!("{} irecv {:?}", get_hostname(), request_lock);

    let send_wr = ibverbs_rs::IbvSendWr::new(
        receiver.out_buffer_mr().addr(),
        receiver.out_buffer_mr().lkey(),
        receiver.in_remote_buffer_addr(),
        receiver.in_remote_buffer_rkey(),
        NcclMetadata::LEN as u64 * n as u64,
        NcclMetadata::LEN as u64 * start_position as u64,
        IbvWrOpcode::RdmaWrite,
        true,
        Some(metadata_request_idx),
        false,
    );
    //println!("{} irecv wr_id {} {:?}", get_hostname(),metadata_request_idx, request_lock);
    let qp = receiver.get_qp(0);
    if let Err(e) = qp.ibv_post_send(send_wr.as_ptr()){
        println!("{} plugin_irecv post error {:?}, {:?}", get_hostname(),e, request_lock);
        log::error!("Error posting send: {:?}", e);
        return 1;
    }
    completion_tracker.mark_uncomplete(metadata_request_idx, 0);
    let test_request = TestRequest{
        qp_list: receiver.qp_list(),
        request_manager: state.request_manager.clone(),
        completion_tracker: state.completion_tracker.clone(),
        wrs_debug: None,
        nccl_metadata: None,
        request_idx: data_request_idx,
        data_tracker: state.data_tracker.clone(),
    };
    let test_request_box = Box::new(test_request);
    let test_request_ptr: *mut TestRequest = Box::into_raw(test_request_box);
    let test_request_ptr_as_c_void: *mut c_void = test_request_ptr as *mut c_void;
    unsafe { * _request = test_request_ptr_as_c_void };
    //std::thread::sleep(std::time::Duration::from_micros(100));
    Box::into_raw(sender_receiver);
    0
}
extern "C" fn plugin_iflush(_recv_comm: *mut c_void, _n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    println!("{} iflush", get_hostname());
    0
}
extern "C" fn plugin_test(mut _request: *mut c_void, _done: *mut c_int, _size: *mut c_int) -> ncclResult_t {
    let test_request_ptr = _request as *mut TestRequest;
    let boxed_test_request = unsafe { Box::from_raw(test_request_ptr) };
    unsafe { * _done = 0; }
    let request_idx = boxed_test_request.request_idx;
    let request_manager = boxed_test_request.request_manager.clone();
    let qp_list = boxed_test_request.qp_list.clone();
    let completion_tracker: Arc<CompletionTracker> = boxed_test_request.completion_tracker.clone();
    {
        let request = request_manager.get_request(request_idx as usize).unwrap();
        let mut request = request.lock().unwrap();
        //println!("{} test {:?}", get_hostname(), request);
        //std::thread::sleep(std::time::Duration::from_secs(1));
        if request.completed && request.completions == 0{
            let sizes = &request.sizes;
            let mut size = 0;
            for s in sizes{
                size += s;
            }
            unsafe { * _size = size as i32; }
            unsafe { * _done = 1; }
            //completion_tracker.mark_complete(request_idx);
            request.reset();
            _request = null_mut();
            return 0;
        }
    }

    for (qp_idx, qp) in qp_list.iter().enumerate(){
        let outstanding_completions = completion_tracker._get_num_of_combined_uncompleted_requests(qp_idx);
        let (wr_list, completed) = match qp.poll_complete(outstanding_completions as usize, IbvWcOpcode::RdmaWrite){
            Ok((completed, wr_id_list)) => {
                (wr_id_list, completed)     
            },
            Err(e) => {
                println!("Error completing: {:?}", e);
                if let Some(wrs_debug) = &boxed_test_request.wrs_debug{
                    println!("wrs_debug {:?}", wrs_debug);
                }
                if let Some(nccl_metadata) = &boxed_test_request.nccl_metadata{
                    println!("nccl_metadata {:?}", nccl_metadata);
                }
                return 1;
            }
        };
        let wr_list_clone = wr_list.clone();
        for (wr_id, imm) in wr_list{
            let is_metadata_completion = if wr_id & (1 << 63) != 0 { true } else { false };
            let request = request_manager.get_request(wr_id as usize).unwrap();
            let mut request = request.lock().unwrap();
            if is_metadata_completion{
                request.md_completed = true;
                completion_tracker.mark_complete(wr_id, qp_idx);
            } else {
                if let RequestType::RecvData = request.request_type{
                    request.sizes.push(imm.unwrap());
                }

                if request.completions > 0 {
                    request.completions -= 1;
                }

                if request.completions == 1{

                    completion_tracker.mark_complete(wr_id, qp_idx);
                    request.completed = true;
                }
            }
            //println!("{} test 2 request qpn {} comp {} {} {:?} {:?}", get_hostname(), qp.qp_num(), completed, wr_id, request, wr_list_clone);
            //std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
    Box::into_raw(boxed_test_request);
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
            maxRecvs: 2,
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
            println!("{} address {} rkey {} lkey {} length {} nreqs {} idx {} ctx {}",
                get_hostname(),
                nccl_metadata.address,
                nccl_metadata.rkey,
                nccl_metadata.lkey,
                nccl_metadata.length,
                nccl_metadata.nreqs,
                nccl_metadata.idx,
                nccl_metadata.context
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
    pub nreqs: u64,
    pub idx: u64,
    pub context: u64,
    pub md_idx: u64,
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
        sender: SenderWrapper,
        state: Arc<State>,
    },
    Receiver{
        receiver: ReceiverWrapper,
        state: Arc<State>,
    }
}

struct SenderWrapper(Arc<Mutex<Box<dyn SenderInterface>>>);

impl SenderWrapper{
    fn new(sender: Box<dyn SenderInterface>) -> Self {
        SenderWrapper(Arc::new(Mutex::new(sender)))
    }
    fn sender(&self) -> Arc<Mutex<Box<dyn SenderInterface>>> {
        self.0.clone()
    }
}
struct ReceiverWrapper(Arc<Mutex<Box<dyn ReceiverInterface>>>);

impl ReceiverWrapper{
    fn new(receiver: Box<dyn ReceiverInterface>) -> Self {
        ReceiverWrapper(Arc::new(Mutex::new(receiver)))
    }
    fn receiver(&self) -> Arc<Mutex<Box<dyn ReceiverInterface>>> {
        self.0.clone()
    }
}
struct RequestManager {
    requests: [Arc<Mutex<Request>>; MAX_REQUESTS as usize], // Array of request objects
    lock: Arc<Mutex<()>>,      // Mutex to synchronize between create_request and complete_request
    index: Arc<AtomicU64>,
}

impl RequestManager {
    fn new() -> Arc<Self> {
        // Initialize requests array and other fields
        let requests = (0..MAX_REQUESTS)
            .map(|_| Arc::new(Mutex::new(Request::new())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(); // Ensure array size matches MAX_REQUESTS

        Arc::new(RequestManager {
            requests,
            lock: Arc::new(Mutex::new(())),
            index: Arc::new(AtomicU64::new(0)),
        })
    }
    fn get_request(&self, idx: usize) -> Option<Arc<Mutex<Request>>> {
        // remove MSB to get the actual index
        let idx = idx & !(1 << 63);
        self.requests.get(idx).cloned()
    }
    /// Returns (data_request_id, metadata_request_id) as a tuple
    fn create_request(&self) -> (u64, u64, Arc<Mutex<Request>>) {
        let _guard = self.lock.lock().unwrap(); // Ensure mutual exclusion
        let current_index = self.index.load(Ordering::SeqCst);
        let metadata_index = (current_index | (1 << 63)) as u64;
        let request = self.requests[current_index as usize].clone();
        
        let next_index = (current_index + 1) % MAX_REQUESTS as u64;
        self.index.store(next_index, Ordering::SeqCst);

        (current_index as u64, metadata_index, request)
    }
}


struct CompletionTracker {
    data_completion_bitmask: Vec<Arc<AtomicUsize>>,       // Bitmask for data request completions
    metadata_completion_bitmask: Arc<AtomicUsize>,   // Bitmask for metadata request completions
}

impl CompletionTracker {
    fn new(num_qps: usize) -> Self {
        let data_completion_bitmask = (0..num_qps)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect::<Vec<_>>();
        CompletionTracker {
            data_completion_bitmask,
            metadata_completion_bitmask: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Mark the request as complete by setting the corresponding bit in the appropriate bitmask
    fn mark_complete(&self, request_id: u64, qp_id: usize) {
        let index = (request_id & !(1 << 63)) as usize; // Get the index without MSB
        let is_metadata = (request_id & (1 << 63)) != 0; // Check if MSB is set

        let mask = 1 << index;

        if is_metadata {
            // Clear the bit in the metadata completion bitmask
            self.metadata_completion_bitmask.fetch_and(!mask, Ordering::SeqCst);
        } else {
            // Clear the bit in the data completion bitmask
            self.data_completion_bitmask[qp_id].fetch_and(!mask, Ordering::SeqCst);
        }
    }

    /// Mark the request as uncomplete by clearing the corresponding bit in the appropriate bitmask
    fn mark_uncomplete(&self, request_id: u64, qp_id: usize) {
        let index = (request_id & !(1 << 63)) as usize; // Get the index without MSB
        let is_metadata = (request_id & (1 << 63)) != 0; // Check if MSB is set

        let mask = 1 << index;

        if is_metadata {
            // Set the bit in the metadata completion bitmask
            self.metadata_completion_bitmask.fetch_or(mask, Ordering::SeqCst);
        } else {
            // Set the bit in the data completion bitmask
            self.data_completion_bitmask[qp_id].fetch_or(mask, Ordering::SeqCst);
        }
    }
        // fn get_num_of_uncompleted_requests counts the bits set in the bitmask to determine the number of uncompleted requests.
        fn get_num_of_uncompleted_data_requests(&self, qp_id: usize) -> usize {
            self.data_completion_bitmask[qp_id]
                .load(Ordering::SeqCst)
                .count_ones() as usize
        }
        fn get_num_of_uncompleted_metadata_requests(&self) -> usize {
            self.metadata_completion_bitmask
                .load(Ordering::SeqCst)
                .count_ones() as usize
        }
        fn _get_num_of_combined_uncompleted_requests(&self, qp_id: usize) -> usize {
            self.data_completion_bitmask[qp_id]
                .load(Ordering::SeqCst)
                .count_ones() as usize + self.metadata_completion_bitmask
                .load(Ordering::SeqCst)
                .count_ones() as usize
        }
}


struct MetadataAllocator {
    tail: Arc<AtomicUsize>, // Atomic tail pointer for tracking the current allocation position
}

impl MetadataAllocator {
    fn new() -> Self {
        MetadataAllocator {
            tail: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Allocates `n` consecutive metadata elements, even with wraparound.
    /// Always succeeds and returns the start index where the allocation began.
    fn allocate(&self, n: usize) -> usize {
        assert!(n <= MAX_REQUESTS as usize, "Cannot allocate more than MAX_REQUESTS elements");

        // Fetch the current tail position
        let start_index = self.tail.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current_tail| {
            // Calculate the next tail position, wrapping around if necessary
            Some((current_tail + n) % MAX_REQUESTS as usize)
        }).unwrap();

        start_index
    }
}

// struct DataTracker tracks the amount of expected data and the actual data received.
struct DataTracker {
    expected_recv: Arc<AtomicU64>, // Expected amount of data
    actual_recv: Arc<AtomicU64>,   // Actual amount of data received
    actual_send: Arc<AtomicU64>   // Actual amount of data sent
}

impl DataTracker {
    fn new() -> Self {
        DataTracker {
            expected_recv: Arc::new(AtomicU64::new(0)),
            actual_recv: Arc::new(AtomicU64::new(0)),
            actual_send: Arc::new(AtomicU64::new(0))
        }
    }

    /// Increments the actual data received by `n` bytes.
    fn increment_actual_recv(&self, n: u64) {
        self.actual_recv.fetch_add(n, Ordering::SeqCst);
    }

    fn increment_expected_recv(&self, n: u64) {
        self.expected_recv.fetch_add(n, Ordering::SeqCst);
    }

    fn increment_actual_send(&self, n: u64) {
        self.actual_send.fetch_add(n, Ordering::SeqCst);
    }

    fn expected_recv(&self) -> u64 {
        self.expected_recv.load(Ordering::SeqCst)
    }

    fn actual_recv(&self) -> u64 {
        self.actual_recv.load(Ordering::SeqCst)
    }
    fn actual_send(&self) -> u64 {
        self.actual_send.load(Ordering::SeqCst)
    }
}

struct TestRequest{
    qp_list: Vec<IbvQp>,
    request_manager: Arc<RequestManager>,
    completion_tracker: Arc<CompletionTracker>,
    wrs_debug: Option<WrsDebug>,
    nccl_metadata: Option<NcclMetadata>,
    request_idx: u64,
    data_tracker: Arc<DataTracker>,
}