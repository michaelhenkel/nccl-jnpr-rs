use std::collections::HashMap;
use std::f32::MAX;
use std::ffi::{c_void, CStr};
use std::fmt::Display;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::os::raw::c_char;
use std::pin::Pin;
use std::ptr::null_mut;
use std::{thread, time};
use std::fmt::Debug;

use ibverbs_rs::receiver::Receiver;
use ibverbs_rs::sender::Sender;
use ibverbs_rs::{print_wr_ids, ControlBufferTrait, Hints, IbvAccessFlags, IbvMr, IbvQp, IbvRecvWr, IbvSendWr, IbvWcOpcode, IbvWrOpcode, LookUpBy, QpMode, SendRecv, SLOT_COUNT};
use rdma_sys::*;
use std::sync::{Arc, Mutex, Once};
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

use libc::{c_int, size_t};

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
struct RequestWrapper(Arc<Mutex<Request>>);

impl RequestWrapper{
    fn new(idx: u32) -> Self{
        RequestWrapper(Arc::new(Mutex::new(Request {
            request_type: RequestType::Available,
            connection_id: 0,
            state_interface: None,
            stage: RequestStage::SendMetadata,
            idx,
            sizes: Vec::new(),
            id: 0,
            completed: false,
            md_completed: false,
        })))
    }
    fn request_type(&self) -> RequestType {
        let request = self.0.lock().unwrap();
        request.request_type.clone()
    }

    fn connection_id(&self) -> u32 {
        let request = self.0.lock().unwrap();
        request.connection_id
    }

    fn set_request_type(&self, request_type: RequestType){
        let mut request = self.0.lock().unwrap();
        request.request_type = request_type;
    }

    fn set_connection_id(&self, connection_id: u32){
        let mut request = self.0.lock().unwrap();
        request.connection_id = connection_id;
    }
    fn set_state_interface(&self, state_interface: StateWrapper){
        let mut request = self.0.lock().unwrap();
        request.state_interface = Some(state_interface);
    }
    fn state_interface(&self) -> StateWrapper{
        let request = self.0.lock().unwrap();
        request.state_interface.clone().unwrap()
    }
    fn stage(&self) -> RequestStage{
        let request = self.0.lock().unwrap();
        request.stage.clone()
    }
    fn set_stage(&self, stage: RequestStage){
        let mut request = self.0.lock().unwrap();
        request.stage = stage;
    }
    fn idx(&self) -> u32{
        let request = self.0.lock().unwrap();
        request.idx
    }
    fn set_sizes(&self, sizes: Vec<u32>){
        let mut request = self.0.lock().unwrap();
        request.sizes = sizes;
    }
    fn sizes(&self) -> Vec<u32>{
        let request = self.0.lock().unwrap();
        request.sizes.clone()
    }
    fn id(&self) -> u64{
        let request = self.0.lock().unwrap();
        request.id
    }
    fn set_id(&self, id: u64){
        let mut request = self.0.lock().unwrap();
        request.id = id;
    }
    fn completed(&self) -> bool{
        let request = self.0.lock().unwrap();
        request.completed
    }
    fn set_completed(&self, completed: bool){
        let mut request = self.0.lock().unwrap();
        request.completed = completed;
    }
    fn md_completed(&self) -> bool{
        let request = self.0.lock().unwrap();
        request.md_completed
    }
    fn set_md_completed(&self, md_completed: bool){
        let mut request = self.0.lock().unwrap();
        request.md_completed = md_completed;
    }
    fn reset(&self){
        let mut request = self.0.lock().unwrap();
        request.request_type = RequestType::Available;
        request.connection_id = 0;
        request.state_interface = None;
        request.stage = RequestStage::SendMetadata;
        request.sizes = Vec::new();
        request.id = 0;
        request.completed = false;
        request.md_completed = false;
    }
}

impl Debug for RequestWrapper{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let request = self.0.lock().unwrap();
        write!(f, "{:?}", request)
    }
}

struct Request{
    request_type: RequestType,
    connection_id: u32,
    state_interface: Option<StateWrapper>,
    stage: RequestStage,
    idx: u32,
    sizes: Vec<u32>,
    id: u64,
    completed: bool,
    md_completed: bool,
}

impl Debug for Request{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Request {{ request_type: {:?}, connection_id: {}, stage: {:?}, idx: {}, sizes: {:?}, id: {}, completed: {}, md_completed: {} }}",
            self.request_type,
            self.connection_id,
            self.stage,
            self.idx,
            self.sizes,
            self.id,
            self.completed,
            self.md_completed
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

trait StateInterface{
    fn get_request(&self, idx: usize) -> Option<RequestWrapper>;
    fn get_interface(self) -> Arc<Box<dyn StateInterface>>;
    fn nccl_md_tail(&self) -> u32;
    fn set_nccl_md_tail(&mut self, nccl_md_tail: u32);
    fn connection_id(&self) -> u32;
    fn id(&self) -> u32;
    fn get_free_request(&self) -> Option<RequestWrapper>;
    fn request_id(&self) -> u32;
    fn set_request_id(&mut self, request_id: u32);
    fn inc_completions(&mut self, completions: u32);
    fn inc_nccl_md_tail(&mut self, n: u32, max_requests: u32);
    fn completions(&self) -> u32;
    fn dec_completions(&mut self, completions: u32);
    fn set_qp(&mut self, qp: IbvQp);
    fn qp(&self) -> IbvQp;
    fn inc_data_sent(&mut self, data_sent: u64);
    fn data_sent(&self) -> u64;
    fn inc_data_sent_completed(&mut self, data_sent_completed: u64);
    fn data_sent_completed(&self) -> u64;
}

#[derive(Clone)]
struct StateWrapper(Arc<Mutex<Box<dyn StateInterface>>>);
impl StateWrapper{
    fn new(state_interface: Box<dyn StateInterface>) -> Self{
        StateWrapper(Arc::new(Mutex::new(state_interface)))
    }
    fn id(&self) -> u32{
        let state_interface = self.0.lock().unwrap();
        state_interface.id()
    }
    fn connection_id(&self) -> u32{
        let state_interface = self.0.lock().unwrap();
        state_interface.connection_id()
    }
    fn nccl_md_tail(&self) -> u32{
        let state_interface = self.0.lock().unwrap();
        state_interface.nccl_md_tail()
    }
    fn set_nccl_md_tail(&mut self, nccl_md_tail: u32){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.set_nccl_md_tail(nccl_md_tail);
    }
    fn get_free_request(&self) -> Option<RequestWrapper>{
        let state_interface = self.0.lock().unwrap();
        state_interface.get_free_request()
    }
    fn request_id(&self) -> u32{
        let state_interface = self.0.lock().unwrap();
        state_interface.request_id()
    }
    fn set_request_id(&mut self, request_id: u32){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.set_request_id(request_id);
    }
    fn inc_nccl_md_tail(&mut self, n: u32, max_requests: u32) {
        let mut state_interface = self.0.lock().unwrap();
        state_interface.inc_nccl_md_tail(n, max_requests);
    }
    fn get_request(&self, idx: usize) -> Option<RequestWrapper>{
        let state_interface = self.0.lock().unwrap();
        state_interface.get_request(idx)
    }
    fn completions(&self) -> u32{
        let state_interface = self.0.lock().unwrap();
        state_interface.completions()
    }
    fn inc_completions(&mut self, completions: u32){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.inc_completions(completions);
    }
    fn dec_completions(&mut self, completions: u32){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.dec_completions(completions);
    }
    fn set_qp(&mut self, qp: IbvQp){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.set_qp(qp);
    }
    fn qp(&self) -> IbvQp{
        let state_interface = self.0.lock().unwrap();
        state_interface.qp()
    }
    fn inc_data_sent(&mut self, data_sent: u64){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.inc_data_sent(data_sent);
    }
    fn data_sent(&self) -> u64{
        let state_interface = self.0.lock().unwrap();
        state_interface.data_sent()
    }
    fn inc_data_sent_completed(&mut self, data_sent_completed: u64){
        let mut state_interface = self.0.lock().unwrap();
        state_interface.inc_data_sent_completed(data_sent_completed);
    }
    fn data_sent_completed(&self) -> u64{
        let state_interface = self.0.lock().unwrap();
        state_interface.data_sent_completed()
    }
}


struct State{
    id: u32,
    connection_id: u32,
    nccl_md_tail: u32,
    recv_send: SendRecv,
    requests: [RequestWrapper; MAX_REQUESTS as usize],
    request_idx: u32,
    request_id: u32,
    sub_request_id: u32,
    completions: u32,
    cts_qp: Option<IbvQp>,
    data_sent: u64,
    data_sent_completed: u64,
}


impl StateInterface for State{
    fn inc_nccl_md_tail(&mut self, n: u32, max_requests: u32) {
        self.nccl_md_tail = (self.nccl_md_tail + n) % max_requests;
    }
    fn get_request(&self, idx: usize) -> Option<RequestWrapper>{
        self.requests.get(idx).map(|request| request.clone())
    }
    fn get_interface(self) -> Arc<Box<dyn StateInterface>>{
        Arc::new(Box::new(self))
    }
    fn nccl_md_tail(&self) -> u32{
        self.nccl_md_tail
    }
    fn set_nccl_md_tail(&mut self, nccl_md_tail: u32){
        self.nccl_md_tail = nccl_md_tail;
    }
    fn connection_id(&self) -> u32{
        self.connection_id
    }
    fn id(&self) -> u32{
        self.id
    }
    fn get_free_request(&self) -> Option<RequestWrapper> {
        for request in self.requests.iter(){
            if let RequestType::Available = request.request_type(){
                return Some(request.clone());
            }
        }
        None
    }
    fn request_id(&self) -> u32{
        self.request_id
    }
    fn set_request_id(&mut self, request_id: u32) {
        self.request_id = request_id;
    }
    fn inc_completions(&mut self, completions: u32) {
        self.completions += completions;
    }
    fn completions(&self) -> u32 {
        self.completions
    }
    fn dec_completions(&mut self, completions: u32) {
        if self.completions >= completions {
            self.completions -= completions;
        }
    }
    fn set_qp(&mut self, qp: IbvQp) {
        self.cts_qp = Some(qp);
    }
    fn qp(&self) -> IbvQp {
        self.cts_qp.clone().unwrap()
    }
    fn data_sent(&self) -> u64 {
        self.data_sent
    }
    fn inc_data_sent(&mut self, data_sent: u64) {
        self.data_sent += data_sent;
    }
    fn data_sent_completed(&self) -> u64 {
        self.data_sent_completed
    }
    fn inc_data_sent_completed(&mut self, data_sent_completed: u64) {
        self.data_sent_completed += data_sent_completed;
    }
}

impl Debug for State{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State {{ id: {}, connection_id: {}, nccl_md_tail: {}, recv_send: {:?}, request_id: {}, sub_request_id: {}, completions: {}, cts_qp: {:?}, data_sent: {}, data_sent_completed: {} }}",
            self.id,
            self.connection_id,
            self.nccl_md_tail,
            self.recv_send,
            self.request_id,
            self.sub_request_id,
            self.completions,
            self.cts_qp,
            self.data_sent,
            self.data_sent_completed
        )
    }
}

impl Default for State{
    fn default() -> Self {
        let mut request_vec = Vec::with_capacity(MAX_REQUESTS as usize);
        for idx in 0..MAX_REQUESTS{
            request_vec.push(RequestWrapper::new(idx));
        }
        let requests: [RequestWrapper; MAX_REQUESTS as usize] = request_vec.try_into().unwrap();
        State{
            id: 0,
            connection_id: 0,
            nccl_md_tail: 0,
            recv_send: SendRecv::Send,
            request_id: 0,
            request_idx: 0,
            sub_request_id: 0,
            completions: 0,
            requests,
            cts_qp: None,
            data_sent: 0,
            data_sent_completed: 0,
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
    state.connection_id = receiver.connection_id();
    state.recv_send = SendRecv::Recv;
    let listen_comm_handle = Box::into_raw(Box::new(SenderReceiver::Receiver { 
        receiver,
        state: StateWrapper::new(Box::new(state)), 
    }));
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
    state.connection_id = sender.connection_id();
    state.recv_send = SendRecv::Send;
    let sender_handle = Box::into_raw(Box::new(SenderReceiver::Sender{
        sender,
        state: StateWrapper::new(Box::new(state)),
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
        let in_buffer_ptr = receiver.in_buffer_ptr();
        let out_buffer_ptr = receiver.out_buffer_ptr();
        let in_buffer_mr = receiver.in_buffer_mr();
        let out_buffer_mr = receiver.out_buffer_mr();
        println!("{} irecv con_id {} in_buffer_ptr {} out_buffer_ptr {} in_buffer_mr {} out_buffer_mr {}",
            get_hostname(),
            receiver.connection_id(),
            in_buffer_ptr as u64,
            out_buffer_ptr as u64,
            in_buffer_mr.addr(),
            out_buffer_mr.addr()
        );
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
            (sender.pd.clone(), "sender".to_string(), state)

        },
        SenderReceiver::Receiver{ref mut receiver, ref mut state} => {
            (receiver.pd.clone(), "recv".to_string(), state)
        }
    };
    let mr = IbvMr::new(pd, data, size, access_flags);
    println!("{} plugin_reg_mr {}, state_id {}, mr addr {}, size {},", get_hostname(),_sender_recv, _state.id(), mr.addr(), size);
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

    let request = state.get_free_request().unwrap();

    state.inc_data_sent(_size as u64);

    request.set_request_type(RequestType::SendData);
    request.set_connection_id(state.connection_id());
    request.set_state_interface(state.clone());
    request.set_stage(RequestStage::WaitForSendCompletion);
    request.set_sizes(vec![_size as u32]);

    let nccl_md_tail = state.nccl_md_tail();

    let start_position = nccl_md_tail % MAX_REQUESTS;

    let in_nccl_metadata_list = sender.in_buffer_ptr() as *mut NcclMetadataList;
    let in_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *in_nccl_metadata_list };
    let in_nccl_metadata = in_nccl_metadata_list.0.get_mut(start_position as usize).unwrap();
    let req_id = in_nccl_metadata.context;
    request.set_id(req_id);
    //println!("{} isend con_id {} in_nccl_metadata {:?}", get_hostname(), state.connection_id(), in_nccl_metadata);

    let mhandle_mr = unsafe { Box::from_raw(_mhandle as *mut IbvMr) };

    let send_wr = ibverbs_rs::IbvSendWr::new(
        _data as u64,
        mhandle_mr.lkey(),
        in_nccl_metadata.address,
        in_nccl_metadata.rkey,
        _size as u64,
        0,
        IbvWrOpcode::RdmaWrite,
        true,
        Some(request.idx() as u64),
        true,
    );

    println!("{} send {:?}", get_hostname(), request);
    thread::sleep(std::time::Duration::from_millis(1));

    if let Err(e) = sender.qp_list[0].ibv_post_send(send_wr.as_ptr()){
        println!("{} plugin_isend post error {:#?}", get_hostname(), e);
        log::error!("Error posting send: {:?}", e);
        return 1;
    }
    state.inc_completions(1);
    state.inc_nccl_md_tail(1, MAX_REQUESTS);
    state.set_qp(sender.qp_list[0].clone());

    let request_box = Box::new(request);
    let request_ptr: *mut RequestWrapper = Box::into_raw(request_box);
    let request_ptr_as_c_void: *mut c_void = request_ptr as *mut c_void;
    unsafe { * _request = request_ptr_as_c_void };

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


    let request = state.get_free_request().unwrap();

    request.set_request_type(RequestType::RecvData);
    request.set_connection_id(state.connection_id());
    request.set_state_interface(state.clone());
    let req_id = rand::random::<u16>();
    request.set_id(req_id as u64);

    let nccl_md_tail = state.nccl_md_tail();
    //state.sub_request_id = rand::random::<u32>();

    let start_position = nccl_md_tail % MAX_REQUESTS;

    let recv_wr = IbvRecvWr::new(None, Some(request.idx() as u64));
    if let Err(e) = receiver.qp_list[0].ibv_post_recv(recv_wr){
        log::error!("Error posting receive: {:?}", e);
        return 1;
    }
    state.inc_completions(1);

    let out_nccl_metadata_list = receiver.out_buffer_ptr() as *mut NcclMetadataList;
    let out_nccl_metadata_list: &mut NcclMetadataList = unsafe { &mut *out_nccl_metadata_list };


    for idx in 0..n {
        let position = (start_position + idx as u32) % MAX_REQUESTS;
        let size = unsafe { * _sizes.offset(idx as isize) };
        let mhandle = unsafe { *mhandles.offset(idx as isize) };
        let mhandle_mr = unsafe { Box::from_raw(mhandle as *mut IbvMr) };
        let data = unsafe { * _data.offset(idx as isize) };
        if data as u64 == mhandle_mr.addr() {
            state.set_request_id(rand::random::<u32>());
        }
        let out_nccl_metadata = out_nccl_metadata_list.0.get_mut(position as usize).unwrap();
        out_nccl_metadata.address = data as u64;
        out_nccl_metadata.rkey = mhandle_mr.rkey();
        out_nccl_metadata.length = size as u64;
        out_nccl_metadata.nreqs = n as u64;
        out_nccl_metadata.idx = request.idx() as u64;
        out_nccl_metadata.context = request.id();
        //println!("{} irecv con_id {} md_tail {} {:?}", get_hostname(), state.connection_id(), state.nccl_md_tail(), out_nccl_metadata);
        //state.insert_request(position as usize, out_nccl_metadata.clone());
        Box::into_raw(mhandle_mr);
    }

    let send_wr = ibverbs_rs::IbvSendWr::new(
        receiver.out_buffer_mr().addr(),
        receiver.out_buffer_mr().lkey(),
        receiver.in_remote_buffer_addr(),
        receiver.in_remote_buffer_rkey(),
        NcclMetadata::LEN as u64 * n as u64,
        NcclMetadata::LEN as u64 * state.nccl_md_tail() as u64,
        IbvWrOpcode::RdmaWrite,
        true,
        Some(request.idx() as u64 + MAX_REQUESTS as u64),
        false,
    );
    //print_wr_ids(send_wr.as_ptr());
    println!("{} recv {:?}", get_hostname(), request);
    thread::sleep(std::time::Duration::from_millis(1));


    if let Err(e) = receiver.qp_list[0].ibv_post_send(send_wr.as_ptr()){
        println!("{} plugin_irecv post error {:#?}", get_hostname(), e);
        log::error!("Error posting send: {:?}", e);
        return 1;
    }

    state.inc_completions(1);
    state.inc_nccl_md_tail(n as u32, MAX_REQUESTS);
    state.set_qp(receiver.qp_list[0].clone());

    let request_box = Box::new(request);
    let request_ptr: *mut RequestWrapper = Box::into_raw(request_box);
    let request_ptr_as_c_void: *mut c_void = request_ptr as *mut c_void;
    unsafe { * _request = request_ptr_as_c_void };

    Box::into_raw(sender_receiver);
    0
}
extern "C" fn plugin_iflush(_recv_comm: *mut c_void, _n: c_int, _data: *mut *mut c_void, _sizes: *mut c_int, _mhandles: *mut *mut c_void, _request: *mut *mut c_void) -> ncclResult_t { 
    0
}
extern "C" fn plugin_test(mut _request: *mut c_void, _done: *mut c_int, _size: *mut c_int) -> ncclResult_t {
    let request_ptr = _request as *mut RequestWrapper;
    let request = unsafe { Box::from_raw(request_ptr) };
    //println!("{} test {:?}", get_hostname(), request);
    thread::sleep(std::time::Duration::from_millis(1));
    unsafe { * _done = 0; }
    let mut state_interface = request.state_interface();

    let mut finish = | | {

        let sizes = request.sizes();
        let mut size = 0;
        for s in sizes{
            size += s;
        }
        /*
        match request.request_type(){
            RequestType::SendData => {
                state_interface.inc_data_sent_completed(size as u64);
                println!("{} test con_id {} data_sent {} data_sent_completed {}", get_hostname(), request.connection_id(), state_interface.data_sent(), state_interface.data_sent_completed());
                thread::sleep(std::time::Duration::from_millis(10));

            },
            RequestType::RecvData => {
                if size == 0 {
                    println!("{} test con_id {} size 0", get_hostname(), request.connection_id());
                    thread::sleep(std::time::Duration::from_millis(10));
                }
            },
            _ => {}
        }
        */
        unsafe { * _size = size as i32; }
        unsafe { * _done = 1; }
        request.reset();
        _request = null_mut();
        return 0;
    };

    match request.request_type(){
        RequestType::SendData => {
            match request.stage(){
                RequestStage::WaitForSendCompletion => {
                    if request.completed(){
                        //println!("{} test {:?} completed", get_hostname(), request);
                        request.set_stage(RequestStage::Finished);
                    } else {
                        //let completions = state_interface.completions();
                        let qp = state_interface.qp();
                        let completions = 1;
                        println!("{} test {:?} waiting for {} completions", get_hostname(), request, completions);
                        match qp.event_complete(completions as usize, IbvWcOpcode::RdmaWrite, None){
                            Ok((completed, wr_id_list)) => {
                                state_interface.dec_completions(completed as u32);
                                if completed > 0 {
                                    //println!("{} test {:?} completed {} wrs {:?}", get_hostname(), request, completed, wr_id_list);
                                    for (wr_id,_imm) in wr_id_list{
                                        let req = state_interface.get_request(wr_id as usize).unwrap();
                                        req.set_completed(true);
                                    }
                                } else {
                                    //println!("{} test {:?} completed {} 0 wrs", get_hostname(), request, completed);
                                }
                                //thread::sleep(std::time::Duration::from_millis(10));
                                
                            },
                            Err(e) => {
                                log::error!("Error completing: {:?}", e);
                                return 1;
                            }
                        }
                    }
                },
                RequestStage::Finished => {
                    finish();
                },
                _ => {
                    println!("{} test {:?} ERROR", get_hostname(), request);
                }
            }
        },
        RequestType::RecvData => {
            match request.stage(){
                RequestStage::SendMetadata => {
                    if request.md_completed(){
                        //println!("{} test {:?} completed", get_hostname(), request);
                        request.set_stage(RequestStage::WaitForData);
                    } else {
                        let completions = state_interface.completions();
                        let qp = state_interface.qp();
                        let completions = 1;
                        println!("{} test {:?} waiting for {} completions", get_hostname(), request, completions);
                        match qp.event_complete(completions as usize, IbvWcOpcode::RdmaWrite, None){
                            Ok((completed, wr_id_list)) => {
                                //println!("{} test {:?} completed {} wrs {:?}", get_hostname(), request, completed, wr_id_list);
                                if completed > 0 {
                                    for (wr_id, imm) in wr_id_list{
                                        if wr_id < MAX_REQUESTS as u64{
                                            let req = state_interface.get_request(wr_id as usize).unwrap();
                                            if let Some(imm) = imm{
                                                req.set_sizes(vec![imm]);
                                            } 
                                            req.set_completed(true);
                                        } else {
                                            let req_idx = (wr_id - MAX_REQUESTS as u64) as usize;
                                            let req = state_interface.get_request(req_idx).unwrap();
                                            req.set_md_completed(true);
                                        }
                                    }
                                    state_interface.dec_completions(completed as u32);
                                }
                                //thread::sleep(std::time::Duration::from_millis(10));
                            },
                            Err(e) => {
                                log::error!("Error completing: {:?}", e);
                                return 1;
                            }
                        }
                    }
                },
                RequestStage::WaitForData => {
                    if request.completed(){
                        //println!("{} test {:?} completed", get_hostname(), request);
                        request.set_stage(RequestStage::Finished);
                    } else {
                        let completions = state_interface.completions();
                        let qp = state_interface.qp();
                        let completions = 1;
                        //println!("{} test {:?} waiting for {} completions", get_hostname(), request, completions);
                        //thread::sleep(std::time::Duration::from_millis(10));
                        match qp.event_complete(completions as usize, IbvWcOpcode::RecvRdmaWithImm, None){
                            Ok((completed, wr_id_list)) => {
                                //println!("{} test {:?} completed {} wrs {:?}", get_hostname(), request, completed, wr_id_list);
                                if completed > 0 {
                                    for (wr_id, imm) in wr_id_list{
                                        if wr_id < MAX_REQUESTS as u64{
                                            let req = state_interface.get_request(wr_id as usize).unwrap();
                                            req.set_completed(true);
                                            if let Some(imm) = imm{
                                                req.set_sizes(vec![imm]);
                                            } 
                                        } else {
                                            let req_idx = (wr_id - MAX_REQUESTS as u64) as usize;
                                            let req = state_interface.get_request(req_idx).unwrap();
                                            req.set_md_completed(true);
                                        }
                                    }
                                    state_interface.dec_completions(completed as u32);
                                }
                                //thread::sleep(std::time::Duration::from_millis(10));
                            },
                            Err(e) => {
                                log::error!("Error completing: {:?}", e);
                                return 1;
                            }
                        }
                    }
                },
                RequestStage::Finished => {
                    finish();
                },
                _ => {
                    println!("{} test {:?} ERROR", get_hostname(), request);
                }
            }
        },
        RequestType::Available => {
            println!("{} test con_id {} available", get_hostname(), request.connection_id());
        }
    }
    Box::into_raw(request);
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
        state: StateWrapper,
    },
    Receiver{
        receiver: Receiver,
        state: StateWrapper,
    }
}