use std::{
    arch::x86_64::{_mm_prefetch, _MM_HINT_T0}, cell::RefCell, char::MAX, ffi::{c_void, CStr, CString}, fmt::{Debug, Display}, fs, mem::MaybeUninit, net::{IpAddr, Ipv4Addr, Ipv6Addr}, os::raw::c_char, pin::Pin, ptr::{self, null_mut}, sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, Once, RwLock
    }
};
use ibverbs_rs::{
    receiver::{Receiver, ReceiverInterface}, sender::{Sender, SenderInterface}, ControlBufferTrait, Family, Hints, IbvAccessFlags, IbvMr, IbvQp, IbvRecvWr, IbvSendWr, IbvWcOpcode, IbvWrOpcode, LookUpBy, QpMode, SendRecv, WrsDebug, SLOT_COUNT
};
use rdma_sys::*;
use env_logger::Env;
mod bindings;
use bindings::*;
use serde::{Serialize, Deserialize};


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

extern "C" fn plugin_init(log_function: ncclDebugLogger_t) -> ncclResult_t {
    unsafe {
        LOG_FUNCTION = log_function;
    }
    initialize_logger();
    0
}
extern "C" fn plugin_devices(ndev: *mut c_int) -> ncclResult_t {
    let filter = match std::env::var("JNPR_NCCL_DEV_FILTER"){
        Ok(filter) => {
            filter.split(",").map(|s| s.to_string()).collect::<Vec<String>>()
        },
        Err(_) => {
            log::error!("Error getting device: {:?}", "JNPR_NCCL_DEV_FILTER not set");
            return 1;
        }
    };
    let num_devices = match get_devices(filter){
        Ok(num_devices) => num_devices,
        Err(e) => {
            log::error!("Error getting devices: {:?}", e);
            return 1;
        }
    };
    unsafe { ndev.write(num_devices as i32) };
    
    0
}

static mut PCI_PATH_STORAGE: Vec<Box<CString>> = Vec::new();

extern "C" fn plugin_get_properties(_dev: c_int, props: *mut ncclNetProperties_v8_t) -> ncclResult_t {
    let filter = match std::env::var("JNPR_NCCL_DEV_FILTER"){
        Ok(filter) => {
            filter.split(",").map(|s| s.to_string()).collect::<Vec<String>>()
        },
        Err(_) => {
            log::error!("Error getting device: {:?}", "JNPR_NCCL_DEV_FILTER not set");
            return 1;
        }
    };
    let device = filter.get(_dev as usize).unwrap();
    match get_device_properties(device.to_string()){
        Ok((_props, pci_path_cstr)) => {
            let boxed_pci_path = Box::new(pci_path_cstr);
            unsafe {
                PCI_PATH_STORAGE.push(boxed_pci_path); // Store the box to keep it alive
            }
            
            unsafe {
                (*props).pciPath = PCI_PATH_STORAGE.last().unwrap().as_ptr() as *mut i8;
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
    let filter = match std::env::var("JNPR_NCCL_DEV_FILTER"){
        Ok(filter) => {
            filter.split(",").map(|s| s.to_string()).collect::<Vec<String>>()
        },
        Err(_) => {
            log::error!("Error getting device: {:?}", "JNPR_NCCL_DEV_FILTER not set");
            return 1;
        }
    };
    let dev = filter.get(_dev as usize).unwrap();
    let num_qps = match std::env::var("JNPR_NCCL_NUM_QPS"){
        Ok(num_qps) => num_qps.parse::<usize>().unwrap(),
        Err(_) => {
            log::error!("Error getting num_qps: {:?}", "JNPR_NCCL_NUM_QPS not set");
            return 1;
        }
    };
    let debug = match std::env::var("JNPR_NCCL_DEBUG"){
        Ok(debug) => debug.parse::<bool>().unwrap(),
        Err(_) => {
            false
        }
    };
    let port = portpicker::pick_unused_port().unwrap();
    let lookup_by = LookUpBy::Name(dev.to_string());
    let qp_mode = QpMode::Multi;
    // 200Gbps in kbps
    let rate_limit = Some(100000000);
    let mut receiver = match Receiver::new::<NcclMetadataList>(lookup_by, port, qp_mode, rate_limit, true){
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
    state.debug = debug;
    let listen_comm_handle = Box::into_raw(Box::new(SenderReceiver::Receiver { 
        receiver: ReceiverWrapper::new(Box::new(receiver)),
        state, 
    }));
    if !listen_comm.is_null() {
        unsafe {
            *listen_comm = listen_comm_handle as *mut c_void;
        }
    }
    0
}
extern "C" fn plugin_connect(_dev: c_int, handle: *mut c_void, send_comm: *mut *mut c_void, _send_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    let filter = match std::env::var("JNPR_NCCL_DEV_FILTER"){
        Ok(filter) => {
            filter.split(",").map(|s| s.to_string()).collect::<Vec<String>>()
        },
        Err(_) => {
            log::error!("Error getting device: {:?}", "JNPR_NCCL_DEV_FILTER not set");
            return 1;
        }
    };
    let dev = filter.get(_dev as usize).unwrap();
    let num_qps = match std::env::var("JNPR_NCCL_NUM_QPS"){
        Ok(num_qps) => num_qps.parse::<usize>().unwrap(),
        Err(_) => {
            log::error!("Error getting num_qps: {:?}", "JNPR_NCCL_NUM_QPS not set");
            return 1;
        }
    };
    let debug = match std::env::var("JNPR_NCCL_DEBUG"){
        Ok(debug) => debug.parse::<bool>().unwrap(),
        Err(_) => {
            false
        }
    };
    let mode: QpMode = match std::env::var("JNPR_NCCL_QPMODE"){
        Ok(mode) => {
            match mode.as_str(){
                "multi" => QpMode::Multi,
                "single" => QpMode::Single,
                _ => {
                    QpMode::Multi
                }
            }
        },
        Err(_) => {
            QpMode::Multi
        }
    };
    let family: Family = match std::env::var("JNPR_NCCL_FAMILY"){
        Ok(family) => {
            match family.as_str(){
                "inet" => Family::Inet,
                "inet6" => Family::Inet6,
                _ => {
                    Family::Inet6
                }
            }
        },
        Err(_) => {
            Family::Inet6
        }
    };
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
    let lookup_by = LookUpBy::Name(dev.to_string());
    let rate_limit = Some(100000000);
    let mut sender = match Sender::new::<NcclMetadataList>(
        lookup_by,
        receiver_address,
        port,
        num_qps as u32,
        family,
        mode,
        rate_limit,
        true,
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
    state.debug = debug;
    state.qps = sender.qps().clone();
    state.in_buffer_ptr = sender.in_buffer_ptr();
    state.qp_health_tracker = sender.qp_health_tracker().clone();
    let sender_handle = Box::into_raw(Box::new(SenderReceiver::Sender{
        sender: SenderWrapper::new(Box::new(sender)),
        state,
    }));

    unsafe {
        *send_comm = sender_handle as *mut c_void;
    }
    0
}
extern "C" fn plugin_accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void, _recv_dev_comm: *mut *mut ncclNetDeviceHandle_v8_t) -> ncclResult_t {
    let mut sender_receiver = unsafe { Box::from_raw(listen_comm as *mut SenderReceiver) };
    if let SenderReceiver::Receiver{ref mut receiver, ref mut state} = *sender_receiver{
        let receiver = receiver.receiver();

        let mut receiver = receiver.write().unwrap();
        
        if let Err(e) = receiver.accept(){
            log::error!("Error accepting: {:?}", e);
            return 1;
        }
        let mut new_state = State::new(receiver.num_qps());
        new_state.id = state.id;
        new_state.connection_id = state.connection_id;
        new_state.recv_send = state.recv_send.clone();
        new_state.debug = state.debug;
        new_state.qps = receiver.qp_list().clone();
        new_state.data_tracker = state.data_tracker.clone();
        new_state.retry_tracker = state.retry_tracker.clone();
        new_state.in_buffer_ptr = receiver.in_buffer_ptr();
        new_state.qp_health_tracker = receiver.qp_health_tracker().clone();
        new_state.metadata_allocator = state.metadata_allocator.clone();
        new_state.request_manager = RequestManager::new();
        new_state.completion_tracker = state.completion_tracker.clone();
        new_state.out_buffer_ptr = receiver.out_buffer_ptr();
        new_state.out_buffer_mr_addr = receiver.out_buffer_mr().addr();
        new_state.out_buffer_mr_lkey = receiver.out_buffer_mr().lkey();
        new_state.in_remote_buffer_addr = receiver.in_remote_buffer_addr();
        new_state.in_remote_buffer_rkey = receiver.in_remote_buffer_rkey();
        new_state.num_qps = receiver.num_qps();
        sender_receiver.set_state(new_state);
        //state.qps = receiver.qp_list().clone();

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
            let sender = sender.sender();
            let sender = match sender.write(){
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
            let receiver = match receiver.write(){
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


const MAX_MESSAGE_SIZE: u64 = 1024 * 1024 * 10;
#[inline(always)]
extern "C" fn plugin_isend(send_comm: *mut c_void, _data: *mut c_void, _size: c_int, _tag: c_int, _mhandle: *mut c_void, mut _request: *mut *mut c_void) -> ncclResult_t { 
    let sender_receiver = unsafe { &mut *(send_comm as *mut SenderReceiver) };
    let (_sender, state) = if let SenderReceiver::Sender { sender, state } = sender_receiver {
        (sender, state)
    } else {
        println!("{} plugin_isend error {:?}", get_hostname(), "Not a sender");
        log::error!("Error accepting: {:?}", "Not a sender");
        return 1;
    };
    let start_position = state.metadata_allocator.allocate(1);
    let in_nccl_metadata_list = state.in_buffer_ptr as *mut NcclMetadataList;
    let in_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *in_nccl_metadata_list };
    let in_nccl_metadata: &mut NcclMetadata = in_nccl_metadata_list.0.get_mut(start_position as usize).unwrap();
    if in_nccl_metadata.address == 0 || in_nccl_metadata.rkey == 0 {
        state.metadata_allocator.deallocate(start_position);
        unsafe { * _request = null_mut() };
        return 0;
    }
    let (mut data_request_idx, _, _heartbeat_idx, request) = state.request_manager.create_request();
    let completion_tracker = state.completion_tracker.clone();
    request.set_idx(data_request_idx as u32);
    request.set_connection_id(state.connection_id);
    request.set_sizes(_size as u64);
    request.set_debug(state.debug);
    request.set_id(in_nccl_metadata.context as u64);
    
    let mhandle_mr = unsafe { &mut *(_mhandle as *mut IbvMr) };

    let mut remote_addr = in_nccl_metadata.address;
    let mut remaining_volume = _size as u64;
    let mut data_addr = _data as u64;
    let qp_list = &state.qps;
    let qps = qp_list.borrow().len() as u64;
    let wrs = &mut state.wrs.borrow_mut();
    let sges = &mut state.sges.borrow_mut();
    let mut wr_idx = 0;
    for (qp_idx, qp) in qp_list.borrow_mut().iter_mut().enumerate() {
        unsafe { _mm_prefetch(qp as *const _ as *const i8, _MM_HINT_T0) };
        request.expected_completions += 1;
        data_request_idx |= (qp_idx as u64 & 0x1F) << 57;
        completion_tracker.mark_uncomplete(data_request_idx);
        let mut last_wr_ptr: *mut ibv_send_wr = std::ptr::null_mut();
        let mut first_wr_ptr: *mut ibv_send_wr = std::ptr::null_mut();
        let mut qp_remaining_volume = (_size as u64 + qps - 1 - qp_idx as u64) / qps;
        while qp_remaining_volume > 0 && remaining_volume > 0 {
            let wr = &mut wrs[wr_idx as usize];
            let sge = &mut sges[wr_idx as usize];
            let message = qp_remaining_volume.min(MAX_MESSAGE_SIZE);
            let is_last_message_for_qp = qp_remaining_volume == message;
            *sge = ibv_sge {
                addr: data_addr,
                length: message as u32,
                lkey: mhandle_mr.lkey(),
            };
            *wr = ibv_send_wr {
                wr_id: data_request_idx,
                next: std::ptr::null_mut(),
                sg_list: sge as *mut ibv_sge,
                num_sge: 1,
                opcode: if is_last_message_for_qp {
                    ibv_wr_opcode::IBV_WR_RDMA_WRITE_WITH_IMM
                } else {
                    ibv_wr_opcode::IBV_WR_RDMA_WRITE
                },
                send_flags: if is_last_message_for_qp {
                    ibv_send_flags::IBV_SEND_SIGNALED.0
                } else {
                    0
                },
                ..unsafe { std::mem::zeroed() } // Zero the rest if necessary
            };
            wr.wr.rdma.remote_addr = remote_addr;
            wr.wr.rdma.rkey = in_nccl_metadata.rkey;
            if is_last_message_for_qp {
                wr.imm_data_invalidated_rkey_union.imm_data = message as u32;
            }
            let wr_ptr = wr as *mut ibv_send_wr;
            if !last_wr_ptr.is_null() {
                unsafe {
                    (*last_wr_ptr).next = wr_ptr;
                }
            } else {
                first_wr_ptr = wr_ptr;
            }
            last_wr_ptr = wr_ptr;

            data_addr += message as u64;
            remote_addr += message as u64;
            remaining_volume -= message;
            qp_remaining_volume -= message;
            wr_idx += 1;
        }
        let _ = qp.ibv_post_send(first_wr_ptr);
    }
    let test_request = TestRequest{
        qp_list,
        request_manager: &mut state.request_manager,
        completion_tracker: state.completion_tracker.clone(),
        wrs_debug: None,
        nccl_metadata: Some(in_nccl_metadata.clone()),
        request_idx: data_request_idx,
        data_tracker: None,
    };

    *in_nccl_metadata = NcclMetadata::default();
    let test_request_box = Box::new(test_request);
    let test_request_ptr = Box::into_raw(test_request_box) as *mut c_void;
    unsafe {
        *_request = test_request_ptr;
    }
    0
}

const MAX_REQUESTS: u32 = 255;
#[inline(always)]
extern "C" fn plugin_irecv(recv_comm: *mut c_void, n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _tags: *mut c_int, mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    let sender_receiver = unsafe { &mut *(recv_comm as *mut SenderReceiver) };
    let (_receiver, state) = if let SenderReceiver::Receiver{ref mut receiver, ref mut state} = *sender_receiver{
        (receiver, state)
    } else {
        log::error!("Error accepting: {:?}", "Not a receiver");
        return 1;
    };
    let num_qps = state.num_qps;
    let qp_list = &state.qps;
    let request_manager = &mut state.request_manager;
    let (mut data_request_idx, mut metadata_request_idx, _heartbeat_idx, request) = request_manager.create_request();
    data_request_idx |= (0 & 0x1F) << 57; 
    metadata_request_idx |= (0 & 0x1F) << 57;
    request.set_connection_id(state.connection_id);
    request.set_idx(data_request_idx as u32);
    let id = rand::random::<u32>();
    request.set_id(id as u64);
    request.set_debug(state.debug);
    request.set_expected_completions(num_qps as u32);

    let metadata_allocator = state.metadata_allocator.clone();
    let start_position = metadata_allocator.allocate(n as usize);
    let completion_tracker = state.completion_tracker.clone();
    for i in 0..num_qps{
        let recv_wr = IbvRecvWr::new(None, Some(data_request_idx));
        if let Err(e) = qp_list.borrow_mut()[i].ibv_post_recv(recv_wr){
            println!("Error posting receive: {:?}", e);
            return 1;
        }
        completion_tracker.mark_uncomplete(data_request_idx);
    }
    let out_nccl_metadata_list = state.out_buffer_ptr as *mut NcclMetadataList;
    let out_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *out_nccl_metadata_list };
    for idx in 0..n {
        let position = start_position + idx as usize;
        let size = unsafe { * _sizes.offset(idx as isize) };
        let mhandle_mr = unsafe { &mut *(*mhandles.offset(idx as isize) as *mut IbvMr) };
        let data = unsafe { * _data.offset(idx as isize) };
        let out_nccl_metadata = out_nccl_metadata_list.0.get_mut(position as usize).unwrap();
        out_nccl_metadata.address = data as u64;
        out_nccl_metadata.rkey = mhandle_mr.rkey();
        out_nccl_metadata.length = size as u64;
        out_nccl_metadata.nreqs = n as u64;
        out_nccl_metadata.idx = request.get_idx() as u64;
        out_nccl_metadata.context = request.get_id() as u64;
        out_nccl_metadata.md_idx = position as u64;
    }
    let send_wr = ibverbs_rs::IbvSendWr::new(
        state.out_buffer_mr_addr,
        state.out_buffer_mr_lkey,
        state.in_remote_buffer_addr,
        state.in_remote_buffer_rkey,
        NcclMetadata::LEN as u64 * n as u64,
        NcclMetadata::LEN as u64 * start_position as u64,
        IbvWrOpcode::RdmaWrite,
        true,
        Some(metadata_request_idx),
        false,
        true
    );

    if let Err(e) = qp_list.borrow_mut()[0].ibv_post_send(send_wr.as_ptr()){
        println!("{} plugin_irecv post error {:?}", get_hostname(),e);
        log::error!("Error posting send: {:?}", e);
        return 1;
    }

    completion_tracker.mark_uncomplete(metadata_request_idx);
    let test_request = TestRequest{
        qp_list,
        request_manager: &mut state.request_manager,
        completion_tracker: state.completion_tracker.clone(),
        wrs_debug: None,
        nccl_metadata: None,
        request_idx: data_request_idx,
        data_tracker: None,
    };
    let test_request_box = Box::new(test_request);
    let test_request_ptr = Box::into_raw(test_request_box) as *mut c_void;
    unsafe {
        *_request = test_request_ptr;
    }
    0
}
extern "C" fn plugin_iflush(_recv_comm: *mut c_void, _n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    println!("{} iflush", get_hostname());
    0
}

const IBV_WC_RECV_RDMA_WITH_IMM: u32 = 129;
const WR_ID_IGNORE_MASK: u64 = 1 << 62;
const WR_ID_METADATA_MASK: u64 = 1 << 63;
const REQUEST_IDX_MASK: u64 = 0x1F << 57;
const MAX_COMPLETIONS: usize = 64;

#[inline(always)]
extern "C" fn plugin_test(mut _request: *mut c_void, _done: *mut c_int, _size: *mut c_int) -> ncclResult_t {
    let boxed_test_request = unsafe { &mut *(_request as *mut TestRequest) };
    {
        let mut request_idx = boxed_test_request.request_idx;
        request_idx &= !(0x1F << 57);
        let request = boxed_test_request.request_manager.get_request(request_idx as usize);
        if request.get_completed() {
            let size: u32 = request.sizes as u32;
            unsafe { *_size = size as i32; }
            unsafe { * _done = 1; }
            request.reset();
            _request = null_mut();
            return 0;
        }
    }
    unsafe { * _done = 0; }
    let completion_tracker = &boxed_test_request.completion_tracker;
    let qp = &mut boxed_test_request.qp_list.borrow_mut()[0];
    let cq = qp.recv_cq().as_ptr();
    let mut wc_array: [MaybeUninit<ibv_wc>; MAX_COMPLETIONS];
    unsafe {
        wc_array = MaybeUninit::uninit().assume_init();
    }
    let wc_ptr = wc_array.as_mut_ptr() as *mut ibv_wc;
    let wc_done = unsafe { ibv_poll_cq(cq, MAX_COMPLETIONS as i32, wc_ptr) };
    let wc_slice = unsafe { std::slice::from_raw_parts(wc_ptr, wc_done as usize) };
    for wc in wc_slice {
        unsafe { _mm_prefetch(wc as *const _ as *const i8, _MM_HINT_T0) };
        if wc.status != ibv_wc_status::IBV_WC_SUCCESS {
            return 1;
        }
        let wr_id = wc.wr_id;
        if wr_id & WR_ID_IGNORE_MASK != 0 {
            continue;
        }
        let is_metadata_completion = (wr_id & WR_ID_METADATA_MASK) != 0;
        let request_idx = (wr_id & !REQUEST_IDX_MASK) as usize;
        let request = boxed_test_request
            .request_manager
            .get_request(request_idx);
        if is_metadata_completion {
            request.set_md_completed(true);
            completion_tracker.mark_complete(wr_id);
            continue;
        }
        if wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM {
            request.sizes += unsafe { wc.imm_data_invalidated_rkey_union.imm_data as u64 };
        }
        request.actual_completions += 1;
        let actual_completions = request.actual_completions;
        let expected_completions = request.expected_completions;
        if actual_completions == expected_completions {
            completion_tracker.mark_complete(wr_id);
            request.set_completed(true);
        }
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
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
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

fn get_devices(filter: Vec<String>) -> anyhow::Result<u32, CustomError>{
    let device_list: *mut *mut ibv_device = unsafe { __ibv_get_device_list(null_mut()) };
    let mut num_devices = 0;
    let mut device_counter = 0;
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
        if filter.contains(&dev_name_string.to_string()) {
            device_counter += 1;
        }
    }
    Ok(device_counter)
}

/*
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs+dev;
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  struct ncclIbDev* ibDev = ncclIbDevs + mergedDev->devs[0];
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

*/

fn knl_module_loaded(path: &str) -> bool {
    fs::metadata(path).is_ok()
}

fn ib_gdr_support_init_once() -> bool {
    knl_module_loaded("/sys/kernel/mm/memory_peers/nv_mem/version") ||
    knl_module_loaded("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
    knl_module_loaded("/sys/module/nvidia_peermem/version")
}

fn get_real_path(device_name: &CStr) -> Option<String> {
    // Construct the PCI device path using the provided device_name (which is a &CStr)
    let pci_path = format!("/sys/class/infiniband/{}/device", device_name.to_str().unwrap());

    // Convert the Rust string to a CString to be compatible with the realpath function
    let pci_path_cstr = CString::new(pci_path).unwrap();

    // Allocate buffer to store the result of realpath (must be at least PATH_MAX in size)
    let mut resolved_path = vec![0 as libc::c_char; libc::PATH_MAX as usize];

    unsafe {
        // Call realpath from libc
        let result = libc::realpath(pci_path_cstr.as_ptr(), resolved_path.as_mut_ptr());

        if result.is_null() {
            // realpath failed, return None
            return None;
        }

        // Convert the resolved path (which is a raw C string) back to a Rust String
        let c_str = CStr::from_ptr(resolved_path.as_ptr());
        let resolved_path_str = c_str.to_string_lossy().into_owned();

        // Return the resolved path as a Rust String
        Some(resolved_path_str)
    }
}

pub fn get_device_properties(dev_name: String) -> anyhow::Result<(ncclNetProperties_t, CString), CustomError> {
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
        let pci_path = get_real_path(device_name).unwrap();
        let pci_path_cstr = CString::new(pci_path).unwrap();
        let speed = 400000;

        let pd = unsafe { ibv_alloc_pd(device_ctx) };
        if pd == null_mut() {
            return Err(CustomError::new("Failed to allocate pd".to_string(), -1));
        }
        let res = unsafe { ibv_reg_dmabuf_mr(pd, 0, 0, 0, -1, 0)};
        let mut dma_buf_supported = true;
        if res == null_mut() {
            dma_buf_supported = false;
        }

        unsafe { ibv_dealloc_pd(pd) };

        let gdr_support = ib_gdr_support_init_once();
        let mut ptr_support = NCCL_PTR_HOST as i32;
        if gdr_support {
            ptr_support |= NCCL_PTR_CUDA as i32;
        }
        if dma_buf_supported {
            ptr_support |= NCCL_PTR_DMABUF as i32;
        }
        let props = ncclNetProperties_t{
            name: device_name.as_ptr() as *mut i8,
            pciPath: pci_path_cstr.as_ptr() as *mut i8,
            guid: device_attr.node_guid,
            ptrSupport: ptr_support,
            speed,
            port: 1,
            maxComms: 1024 * 1024,
            regIsGlobal: 1,
            latency: 0.0,
            maxRecvs: 8,
            netDeviceType: 0,
            netDeviceVersion: 0,
        };
        return Ok((props, pci_path_cstr));
    
    }
    Err(CustomError::new("Device not found".to_string(), -1))

}

pub fn get_device_properties_by_index(dev_idx: i32) -> anyhow::Result<ncclNetProperties_v8_t, CustomError> {
    let device_list: *mut *mut ibv_device = unsafe { __ibv_get_device_list(null_mut()) };
    let mut num_devices = 0;
    while !unsafe { *device_list.offset(num_devices) }.is_null() {
        num_devices += 1;
    }
    if num_devices == 0 {
        return Err(CustomError::new("ibv_get_device_list".to_string(), -1).into());
    }
    let device: *mut ibv_device = unsafe { *device_list.offset(dev_idx as isize) };
    let device_ctx = unsafe { ibv_open_device(device) };
    if device_ctx == null_mut() {
        return Err(CustomError::new("Failed to open device".to_string(), -1));
    }
    let mut device_attr = unsafe { std::mem::zeroed::<ibv_device_attr>() };
    let ret = unsafe { ibv_query_device(device_ctx, &mut device_attr) };
    if ret != 0 {
        return Err(CustomError::new("Failed to query device".to_string(), ret));
    }
    let device_name = unsafe { CStr::from_ptr((*device).name.as_ptr()) };
    let pci_path = format!("/sys/class/infiniband/{}/device", device_name.to_str().unwrap());
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
        maxRecvs: 8,
        netDeviceType: 0,
        netDeviceVersion: 0,
    };
    return Ok(props);

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
        state: State,
    },
    Receiver{
        receiver: ReceiverWrapper,
        state: State,
    }
}

impl SenderReceiver{
    fn set_state(&mut self, state: State){
        match self{
            SenderReceiver::Sender{state: ref mut s, ..} => {
                *s = state;
            },
            SenderReceiver::Receiver{state: ref mut s, ..} => {
                *s = state;
            }
        }
    }
}

struct SenderWrapper(Arc<RwLock<Box<dyn SenderInterface>>>);

impl SenderWrapper{
    fn new(sender: Box<dyn SenderInterface>) -> Self {
        SenderWrapper(Arc::new(RwLock::new(sender)))
    }
    fn sender(&self) -> Arc<RwLock<Box<dyn SenderInterface>>> {
        self.0.clone()
    }
}
struct ReceiverWrapper(Arc<RwLock<Box<dyn ReceiverInterface>>>);

impl ReceiverWrapper{
    fn new(receiver: Box<dyn ReceiverInterface>) -> Self {
        ReceiverWrapper(Arc::new(RwLock::new(receiver)))
    }
    fn receiver(&self) -> Arc<RwLock<Box<dyn ReceiverInterface>>> {
        self.0.clone()
    }
}

struct Request{
    connection_id: u32,
    idx: u32,
    sizes: u64,
    id: u64,
    completed: bool,
    md_completed: bool,
    md_idx: u32,
    expected_completions: u32,
    actual_completions: u32,
    debug: bool,
    retries: u32,
}

impl Request{
    fn new(idx: u32, conn_id: u32, size: u64, debug: bool, id: u64) -> Self {
        Request {
            connection_id: conn_id,
            idx: idx,
            sizes: size,
            id: id,
            completed: false,
            md_completed: false,
            md_idx: 0,
            expected_completions: 0,
            actual_completions: 0,
            debug: debug,
            retries: 0,
        }
    }
    
    fn reset(&mut self) {
        self.connection_id = 0;
        self.idx = 0;
        self.sizes = 0;
        self.id = 0;
        self.completed = false;
        self.md_completed = false;
        self.md_idx = 0;
        self.expected_completions = 0;
        self.actual_completions = 0;
        self.debug = false;
        self.retries = 0;
    }
    
    #[inline(always)]
    fn set_idx(&mut self, idx: u32) {
        self.idx = idx;
    }
    
    #[inline(always)]
    fn set_connection_id(&mut self, connection_id: u32) {
        self.connection_id = connection_id;
    }
    
    #[inline(always)]
    fn set_sizes(&mut self, sizes: u64) {
        self.sizes = sizes;
    }
    
    #[inline(always)]
    fn set_id(&mut self, id: u64) {
        self.id = id;
    }
    
    #[inline(always)]
    fn set_completed(&mut self, completed: bool) {
        self.completed = completed;
    }
    
    #[inline(always)]
    fn set_md_completed(&mut self, md_completed: bool) {
        self.md_completed = md_completed;
    }
    
    #[inline(always)]
    fn set_expected_completions(&mut self, expected_completions: u32) {
        self.expected_completions = expected_completions;
    }
    
    #[inline(always)]
    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    
    #[inline(always)]
    fn get_idx(&self) -> u32 {
        self.idx
    }
    
    #[inline(always)]
    fn get_id(&self) -> u64 {
        self.id
    }
    
    #[inline(always)]
    fn get_completed(&self) -> bool {
        self.completed
    }
}

impl Default for Request{
    fn default() -> Self {
        Request{
            connection_id: 0,
            idx: 0,
            sizes: 0,
            id: 0,
            completed: false,
            md_completed: false,
            md_idx: 0,
            expected_completions: 0,
            actual_completions: 0,
            debug: false,
            retries: 0,
        }
    }
}

impl Debug for Request{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {{ connection_id: {:?}, idx: {}, sizes: {}, id: {}, completed: {}, md_completed: {}, compl: {}, actual_compl: {}, debug: {}, retries: {} }}",
            self.connection_id,
            self.idx,
            self.sizes,
            self.id,
            self.completed,
            self.md_completed,
            self.expected_completions,
            self.actual_completions,
            self.debug,
            self.retries
        )
    }
}

#[derive(Debug, Clone)]
enum RequestStage{
    SendMetadata,
    WaitForData,
    WaitForSendCompletion,
}

#[derive(Debug, Default, Clone)]
enum RequestType{
    SendData,
    RecvData,
    #[default]
    Available,
}

struct RequestManager {
    requests: [Request; MAX_REQUESTS as usize],
    //lock: Arc<Mutex<()>>,
    index: u64,
}

impl RequestManager {
    fn new() -> Self {
        let requests: [Request; MAX_REQUESTS as usize] = (0..MAX_REQUESTS)
            .map(|_| Request::default())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        RequestManager {
            requests,
            //lock: Arc::new(Mutex::new(())),
            index: 0,
        }
    }
    fn get_request(&mut self, idx: usize) -> &mut Request {
        let idx = idx & !(1 << 63);
        &mut self.requests[idx as usize]
    }
    fn create_request(&mut self) -> (u64, u64, u64, &mut Request) {
        //let _guard = self.lock.lock().unwrap(); // Ensure mutual exclusion
        let current_index = self.index;
        let metadata_index = (current_index | (1 << 63)) as u64;
        let heartbeat_index = (current_index | (1 << 62)) as u64;
        let request = &mut self.requests[current_index as usize];
        let next_index = (current_index + 1) % MAX_REQUESTS as u64;
        self.index = next_index;
        (current_index as u64, metadata_index, heartbeat_index, request)
    }
}

struct RetryTracker {
    retries: Arc<AtomicUsize>,
}
impl RetryTracker{
    fn new() -> Self {
        RetryTracker{
            retries: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn increment(&self){
        self.retries.fetch_add(1, Ordering::SeqCst);
    }
    fn get(&self) -> usize {
        self.retries.load(Ordering::SeqCst)
    }
    fn reset(&self){
        self.retries.store(0, Ordering::SeqCst);
    }
}

struct CompletionTracker {
    data_completion_bitmask: Arc<AtomicUsize>,       // Bitmask for data request completions
    metadata_completion_bitmask: Arc<AtomicUsize>,   // Bitmask for metadata request completions
    heartbeat_completion_bitmask: Arc<AtomicUsize>,  // Bitmask for heartbeat request completions
}

impl CompletionTracker {
    fn new() -> Self {
        CompletionTracker {
            data_completion_bitmask: Arc::new(AtomicUsize::new(0)),
            metadata_completion_bitmask: Arc::new(AtomicUsize::new(0)),
            heartbeat_completion_bitmask: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Mark the request as complete by setting the corresponding bit in the appropriate bitmask
    fn mark_complete(&self, request_id: u64) {
        let is_metadata = (request_id & (1 << 63)) != 0;   // Check if MSB is set
        let is_heartbeat = (request_id & (1 << 62)) != 0;  // Check if 2nd MSB is set

        if is_metadata {
            let index = (request_id & 0x3F) as usize;      // Bits 0-5
            let mask = 1 << index;
            self.metadata_completion_bitmask.fetch_and(!mask, Ordering::SeqCst);
        } else if is_heartbeat {
            let index = (request_id & 0x3F) as usize;      // Bits 0-5
            let mask = 1 << index;
            self.heartbeat_completion_bitmask.fetch_and(!mask, Ordering::SeqCst);
        } else {
            let qp_id = ((request_id >> 59) & 0x7) as usize;   // Bits 59-61
            let index = (request_id & 0x3F) as usize;          // Bits 0-5

            let bit_index = (qp_id << 6) | index;  // Combine qp_id and index
            let mask = 1 << bit_index;
            self.data_completion_bitmask.fetch_and(!mask, Ordering::SeqCst);
        }
    }

    /// Mark the request as uncomplete by clearing the corresponding bit in the appropriate bitmask
    fn mark_uncomplete(&self, request_id: u64) {
        let is_metadata = (request_id & (1 << 63)) != 0;
        let is_heartbeat = (request_id & (1 << 62)) != 0;

        if is_metadata {
            let index = (request_id & 0x3F) as usize;
            let mask = 1 << index;
            self.metadata_completion_bitmask.fetch_or(mask, Ordering::SeqCst);
        } else if is_heartbeat {
            let index = (request_id & 0x3F) as usize;
            let mask = 1 << index;
            self.heartbeat_completion_bitmask.fetch_or(mask, Ordering::SeqCst);
        } else {
            let qp_id = ((request_id >> 59) & 0x7) as usize;
            let index = (request_id & 0x3F) as usize;

            let bit_index = (qp_id << 6) | index;
            let mask = 1 << bit_index;
            self.data_completion_bitmask.fetch_or(mask, Ordering::SeqCst);
        }
    }
    fn get_num_of_combined_uncompleted_requests(&self) -> usize {
        self.data_completion_bitmask
            .load(Ordering::SeqCst)
            .count_ones() as usize
            + self.metadata_completion_bitmask
                .load(Ordering::SeqCst)
                .count_ones() as usize
            + self.heartbeat_completion_bitmask
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
    fn deallocate(&self, start_index: usize) {
        self.tail.store(start_index, Ordering::SeqCst);
    }
}

// struct DataTracker tracks the amount of expected data and the actual data received.
struct DataTracker {
    expected_recv: Arc<AtomicU64>, // Expected amount of data
    actual_recv: Arc<AtomicU64>,   // Actual amount of data received
    actual_send: Arc<AtomicU64>   // Actual amount of data sent
}

#[allow(dead_code)]
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

#[allow(dead_code)]
struct TestRequest<'a>{
    qp_list: &'a RefCell<Vec<IbvQp>>,
    request_manager: &'a mut RequestManager,
    completion_tracker: Arc<CompletionTracker>,
    wrs_debug: Option<WrsDebug>,
    nccl_metadata: Option<NcclMetadata>,
    request_idx: u64,
    data_tracker: Option<Arc<DataTracker>>,
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

const MAX_QPS: usize = 64;
const MAX_WRS: usize = MAX_QPS * 64;

#[allow(dead_code)]
struct State{
    id: u32,
    connection_id: u32,
    recv_send: SendRecv,
    request_manager: RequestManager,
    metadata_allocator: Arc<MetadataAllocator>,
    completion_tracker: Arc<CompletionTracker>,
    data_tracker: Arc<DataTracker>,
    retry_tracker: Arc<RetryTracker>,
    debug: bool,
    retries: u32,
    in_buffer_ptr: *mut c_void,
    qps: RefCell<Vec<IbvQp>>,
    qp_health_tracker: Arc<AtomicU32>,
    out_buffer_ptr: *mut c_void,
    out_buffer_mr_addr: u64,
    out_buffer_mr_lkey: u32,
    in_remote_buffer_addr: u64,
    in_remote_buffer_rkey: u32,
    num_qps: usize,
    wrs: RefCell<[ibv_send_wr; MAX_WRS]>,
    sges: RefCell<[ibv_sge; MAX_WRS]>,
    total_size: Arc<AtomicU64>,
}
impl State{
    fn new(num_qps: usize) -> Self{
        let wrs: [ibv_send_wr; MAX_WRS] = unsafe { std::mem::zeroed() };
        let sges: [ibv_sge; MAX_WRS] = unsafe { std::mem::zeroed() };
        State{
            id: 0,
            connection_id: 0,
            recv_send: SendRecv::Send,
            request_manager: RequestManager::new(),
            metadata_allocator: Arc::new(MetadataAllocator::new()),
            completion_tracker: Arc::new(CompletionTracker::new()),
            data_tracker: Arc::new(DataTracker::new()),
            retry_tracker: Arc::new(RetryTracker::new()),
            debug: false,
            retries: 0,
            in_buffer_ptr: null_mut(),
            qps: RefCell::new(Vec::new()),
            qp_health_tracker: Arc::new(AtomicU32::new(0)),
            out_buffer_ptr: null_mut(),
            out_buffer_mr_addr: 0,
            out_buffer_mr_lkey: 0,
            in_remote_buffer_addr: 0,
            in_remote_buffer_rkey: 0,
            num_qps,
            wrs: RefCell::new(wrs),
            sges: RefCell::new(sges),
            total_size: Arc::new(AtomicU64::new(0)),
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