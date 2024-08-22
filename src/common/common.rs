use std::{collections::{BTreeMap, HashMap}, f32::consts::E, ffi::{c_void, CStr}, fmt::Display, fs, io::{self, Read, Write}, net::{IpAddr, Ipv4Addr, Ipv6Addr, TcpListener, TcpStream}, path::PathBuf, ptr::{self, null_mut}};
use libc::c_int;
use rdma_sys::*;

use crate::{bindings::{NCCL_PTR_CUDA, NCCL_PTR_DMABUF, NCCL_PTR_HOST}, ncclNetProperties_v8_t, NcclNetSocketHandle, QpList, RemoteHandle, SocketComm};

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

pub fn get_gid(context: *mut ibv_context, port_num: u8) -> ibv_gid {
    let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
    unsafe { ibv_query_gid(context, port_num, 1, &mut gid) };
    gid
}

pub fn get_port_info(context: *mut ibv_context, port_num: u8) -> anyhow::Result<ibv_port_attr, CustomError> {
    let mut port_attr: ibv_port_attr = unsafe { std::mem::zeroed() };
    let ret = unsafe { ___ibv_query_port(context, port_num, &mut port_attr) };
    if ret != 0 {
        return Err(CustomError::new("ibv_query_port".to_string(), ret).into());
    }
    Ok(port_attr)
}

pub fn gid_to_ipv6_string(gid: ibv_gid) -> Option<Ipv6Addr> {
    unsafe {
        // Access the raw bytes of the gid union
        let raw_gid = gid.raw;
        // check if all bytes are zero
        let mut all_zero = true;
        for i in 0..16{
            if raw_gid[i] != 0{
                all_zero = false;
                break;
            }
        }
        if all_zero{
            return None;
        }

        // Create an Ipv6Addr from the raw bytes
        let ipv6_addr = Ipv6Addr::new(
            (raw_gid[0] as u16) << 8 | (raw_gid[1] as u16),
            (raw_gid[2] as u16) << 8 | (raw_gid[3] as u16),
            (raw_gid[4] as u16) << 8 | (raw_gid[5] as u16),
            (raw_gid[6] as u16) << 8 | (raw_gid[7] as u16),
            (raw_gid[8] as u16) << 8 | (raw_gid[9] as u16),
            (raw_gid[10] as u16) << 8 | (raw_gid[11] as u16),
            (raw_gid[12] as u16) << 8 | (raw_gid[13] as u16),
            (raw_gid[14] as u16) << 8 | (raw_gid[15] as u16),
        );

        // Convert the Ipv6Addr to a string
        Some(ipv6_addr)
    }
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


pub fn get_listen_address(dev_name: String) -> anyhow::Result<Option<Ipv6Addr>, CustomError>{
    let device_list: *mut *mut ibv_device = unsafe { __ibv_get_device_list(null_mut()) };
    let mut num_devices = 0;
    while !unsafe { *device_list.offset(num_devices) }.is_null() {
        num_devices += 1;
    }
    if num_devices == 0 {
        return Err(CustomError::new("ibv_get_device_list".to_string(), -1).into());
    }
    for i in 0..num_devices {
        let device = unsafe { *device_list.offset(i as isize) };
        let device_ctx = unsafe { ibv_open_device(device) };
        if device_ctx == null_mut() {
            return Err(CustomError::new("Failed to open device".to_string(), -1));
        }
        let device_name = unsafe { CStr::from_ptr((*device).name.as_ptr()) };
        if dev_name.as_str() != device_name.to_str().unwrap(){
            continue;
        }
        let mut device_attr: ibv_device_attr = unsafe { std::mem::zeroed::<ibv_device_attr>() };
        let ret = unsafe { ibv_query_device(device_ctx, &mut device_attr) };
        if ret != 0 {
            return Err(CustomError::new("Failed to query device".to_string(), ret));
        }
        let num_ports = device_attr.phys_port_cnt;
        for i in 1..=num_ports {
            let mut port_attr: ibv_port_attr = unsafe { std::mem::zeroed::<ibv_port_attr>() };
            let ret = unsafe { ___ibv_query_port(device_ctx, i, &mut port_attr) };
            if ret != 0 {
                return Err(CustomError::new("Failed to query port".to_string(), ret));
            }
            let gid_tbl_len = port_attr.gid_tbl_len;
            for j in 0..gid_tbl_len {
                let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
                unsafe { ibv_query_gid(device_ctx, i, j, &mut gid) };
                let address = if let Some(gid_v6) = gid_to_ipv6_string(gid){
                    match gid_v6.to_ipv4(){
                        Some(gid_v4) => {
                            Some(IpAddr::V4(gid_v4))
                        },
                        None => {
                            let segments = gid_v6.segments();
                            if (segments[0] & 0xffc0) != 0xfe80 {
                                return Ok(Some(gid_v6));
                            } else {
                                None
                            }
                        },
                    }
                } else {
                    None
                        
                };
                if let Some(address) = address{
                    match read_gid_type(device_name.to_str().unwrap(), i, j)?{
                        GidType::ROCEv2 => {


                        },
                        GidType::RoCEv1 => {

                        },
                    }
                }
            }
        }
    }
    Ok(None)
}

pub fn generate_gid_table(dev_name: Option<String>) -> anyhow::Result<GidTable,CustomError>{

    let mut gid_table = GidTable{
        context: IbvContext(null_mut()),
        v4_table: BTreeMap::new(),
        v6_table: BTreeMap::new(),
    };
    let device_list: *mut *mut ibv_device = unsafe { __ibv_get_device_list(null_mut()) };

    // get the number of elements in the list
    let mut num_devices = 0;
    while !unsafe { *device_list.offset(num_devices) }.is_null() {
        num_devices += 1;
    }
    if num_devices == 0 {
        return Err(CustomError::new("ibv_get_device_list".to_string(), -1).into());
    }
    for i in 0..num_devices {
        let device = unsafe { *device_list.offset(i as isize) };
        let device_ctx = unsafe { ibv_open_device(device) };
        if device_ctx == null_mut() {
            return Err(CustomError::new("Failed to open device".to_string(), -1));
        }
        let device_name = unsafe { CStr::from_ptr((*device).name.as_ptr()) };
        if let Some(dev_name) = dev_name.clone() {
            if dev_name.as_str() != device_name.to_str().unwrap(){
                continue;
            }
        }
        gid_table.context = IbvContext(device_ctx);
        let mut device_attr: ibv_device_attr = unsafe { std::mem::zeroed::<ibv_device_attr>() };
        let ret = unsafe { ibv_query_device(device_ctx, &mut device_attr) };
        if ret != 0 {
            return Err(CustomError::new("Failed to query device".to_string(), ret));
        }
        let num_ports = device_attr.phys_port_cnt;
        for i in 1..=num_ports {
            let mut port_attr: ibv_port_attr = unsafe { std::mem::zeroed::<ibv_port_attr>() };
            let ret = unsafe { ___ibv_query_port(device_ctx, i, &mut port_attr) };
            if ret != 0 {
                return Err(CustomError::new("Failed to query port".to_string(), ret));
            }
            let gid_tbl_len = port_attr.gid_tbl_len;
            for j in 0..gid_tbl_len {
                let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
                unsafe { ibv_query_gid(device_ctx, i, j, &mut gid) };
                let address = if let Some(gid_v6) = gid_to_ipv6_string(gid){
                    match gid_v6.to_ipv4(){
                        Some(gid_v4) => {
                            Some(IpAddr::V4(gid_v4))
                        },
                        None => {
                            let segments = gid_v6.segments();
                            if (segments[0] & 0xffc0) != 0xfe80 {
                                Some(IpAddr::V6(gid_v6))
                            } else {
                                None
                            }
                        },
                    }
                } else {
                    None
                        
                };
                if let Some(address) = address{
                    match read_gid_type(device_name.to_str().unwrap(), i, j)?{
                        GidType::ROCEv2 => {
                            let gid = Box::new(gid);
                            let gid_ptr = Box::into_raw(gid);
                            let gid_entry = GidEntry{
                                gid: IbvGid(gid_ptr),
                                port: i,
                                gidx: j,
                            };
                            gid_table.add_entry(gid_entry, address);

                        },
                        GidType::RoCEv1 => {

                        },
                    }
                }
            }
        }
    }
    Ok(gid_table)
}

fn read_gid_type(device_name: &str, port: u8, gid_index: i32) -> anyhow::Result<GidType, CustomError> {
    // Construct the file path
    let path = PathBuf::from(format!(
        "/sys/class/infiniband/{}/ports/{}/gid_attrs/types/{}",
        device_name, port, gid_index
    ));

    // Read the file contents
    let gid_type = fs::read_to_string(path).map_err(|e| CustomError::new(e.to_string(), -1))?;

    // Return the contents as a String
    let gid_type = GidType::from_str(gid_type.trim());
    Ok(gid_type)
}

#[derive(Clone)]
pub struct IbvGid(pub *mut ibv_gid);
unsafe impl Send for IbvGid{}
unsafe impl Sync for IbvGid{}
impl IbvGid{
    pub fn ibv_gid(&self) -> *mut ibv_gid{
        self.0
    }
}

#[derive(Clone)]
pub struct IbvContext(pub *mut ibv_context);
unsafe impl Send for IbvContext{}
unsafe impl Sync for IbvContext{}
impl IbvContext{
    pub fn new(device_name: &CStr) -> anyhow::Result<IbvContext, CustomError> {
        let device_list: *mut *mut ibv_device = unsafe { ibv_get_device_list(null_mut()) };
        if device_list.is_null() {
            return Err(CustomError::new("ibv_get_device_list".to_string(), -1).into());
        }
        let mut device: *mut ibv_device = null_mut();
        let mut context: *mut ibv_context = null_mut();
        let mut found = false;
        let mut i = 0;
        while !device_list.is_null() {
            device = unsafe { *device_list.wrapping_add(i) };
            if device.is_null() {
                break;
            }
            let name: &CStr = unsafe { CStr::from_ptr((*device).name.as_ptr()) }; // Convert array to raw pointer
            if name == device_name {
                found = true;
                break;
            }
            i += 1;
        }
        if found {
            context = unsafe { ibv_open_device(device) };
        }
        unsafe { ibv_free_device_list(device_list) };
        Ok(IbvContext(context))
    }

    pub fn ibv_context(&self) -> *mut ibv_context{
        self.0
    }
}

#[derive(Clone)]
pub struct GidTable{
    pub context: IbvContext,
    pub v4_table: BTreeMap<Ipv4Addr, GidEntry>,
    pub v6_table: BTreeMap<Ipv6Addr, GidEntry>,
}

impl GidTable{
    pub fn add_entry(&mut self, gid_entry: GidEntry, address: IpAddr){
        match address{
            IpAddr::V4(v4) => {
                self.v4_table.insert(v4, gid_entry);
            },
            IpAddr::V6(v6) => {
                self.v6_table.insert(v6, gid_entry);
            }
        }
    }
}

#[derive(Clone)]
pub struct GidEntry{
    pub gid: IbvGid,
    pub port: u8,
    pub gidx: i32,
}

#[derive(Clone)]
pub enum Family{
    V4,
    V6
}

enum GidType{
    ROCEv2,
    RoCEv1,
}
impl GidType{
    fn from_str(s: &str) -> GidType{
        match s{
            "RoCE v2" => GidType::ROCEv2,
            "IB/RoCE v1" => GidType::RoCEv1,
            _ => GidType::ROCEv2,
        }
    }
}

pub fn create_context(device_name: &CStr) -> *mut ibv_context {
    let device_list: *mut *mut ibv_device = unsafe { ibv_get_device_list(null_mut()) };
    if device_list.is_null() {
        return null_mut();
    }
    let mut device: *mut ibv_device = null_mut();
    let mut context: *mut ibv_context = null_mut();
    let mut found = false;
    let mut i = 0;
    while !device_list.is_null() {
        device = unsafe { *device_list.wrapping_add(i) };
        if device.is_null() {
            break;
        }
        let name: &CStr = unsafe { CStr::from_ptr((*device).name.as_ptr()) }; // Convert array to raw pointer
        if name == device_name {
            found = true;
            break;
        }
        i += 1;
    }
    if found {
        context = unsafe { ibv_open_device(device) };
    }
    unsafe { ibv_free_device_list(device_list) };
    context
}

pub fn create_event_channel(context: *mut ibv_context) -> anyhow::Result<EventChannel, CustomError> {
    let channel = unsafe { ibv_create_comp_channel(context) };
    if channel.is_null() {
        return Err(CustomError::new("ibv_create_comp_channel".to_string(), -1).into());
    }
    Ok(EventChannel(channel))
}

pub fn create_protection_domain(context: *mut ibv_context) -> anyhow::Result<ProtectionDomain, CustomError> {
    let pd = unsafe { ibv_alloc_pd(context) };
    if pd == null_mut() {
        return Err(CustomError::new("ibv_alloc_pd".to_string(), -1).into());
    }
    Ok(ProtectionDomain(pd))
}

pub fn create_create_completion_queue(context: *mut ibv_context, channel: *mut ibv_comp_channel) -> *mut ibv_cq {
    unsafe { ibv_create_cq(context, 4096, null_mut(), channel, 0) }
}

pub fn create_queue_pair(
    pd: *mut ibv_pd,
    cq: *mut ibv_cq,
) -> *mut ibv_qp {
    let mut qp_init_attr = ibv_qp_init_attr {
        qp_context: null_mut(),
        send_cq: cq,
        recv_cq: cq,
        srq: null_mut(),
        cap: ibv_qp_cap {
            max_send_wr: 4096,
            max_recv_wr: 4096,
            max_send_sge: 15,
            max_recv_sge: 15,
            max_inline_data: 64,
        },
        qp_type: ibv_qp_type::IBV_QPT_RC,
        sq_sig_all: 0,
    };
    unsafe { ibv_create_qp(pd, &mut qp_init_attr) }
}

pub fn set_qp_init_state(qp: *mut ibv_qp, port: u8) -> anyhow::Result<(),CustomError> {
    let mut qp_attr = unsafe { std::mem::zeroed::<ibv_qp_attr>() };
    qp_attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port;
    qp_attr.qp_access_flags = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0 | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0 | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0;
    let qp_attr_mask = ibv_qp_attr_mask::IBV_QP_STATE | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX | ibv_qp_attr_mask::IBV_QP_PORT | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
    let ret = unsafe { ibv_modify_qp(qp, &mut qp_attr, qp_attr_mask.0 as i32) };
    if ret != 0 {
        return Err(CustomError::new("ibv_modify_qp".to_string(), ret).into());
    }
    Ok(())
}

pub fn connect_qp(qp: *mut ibv_qp, gid: ibv_gid, qpn: u32, psn: u32, my_psn: u32, gidx: i32) -> anyhow::Result<(), CustomError> {
    let mut qp_attr = unsafe { std::mem::zeroed::<ibv_qp_attr>() };
    qp_attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
    qp_attr.path_mtu = ibv_mtu::IBV_MTU_4096;
    qp_attr.dest_qp_num = qpn;
    qp_attr.rq_psn = psn;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;

    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = 1;
    qp_attr.ah_attr.dlid = 0;
    qp_attr.ah_attr.grh.dgid = gid;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.sgid_index = gidx as u8;
    qp_attr.ah_attr.grh.hop_limit = 10;

    let qp_attr_mask = 
        ibv_qp_attr_mask::IBV_QP_STATE |
        ibv_qp_attr_mask::IBV_QP_AV |
        ibv_qp_attr_mask::IBV_QP_PATH_MTU |
        ibv_qp_attr_mask::IBV_QP_DEST_QPN |
        ibv_qp_attr_mask::IBV_QP_RQ_PSN |
        ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC |
        ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
    let ret = unsafe { ibv_modify_qp(qp, &mut qp_attr, qp_attr_mask.0 as i32) };
    if ret != 0 {
        println!("remote gid: {}", gid_to_ipv6_string(gid).unwrap());
        println!("qp_attr.dest_qp_num: {}", qp_attr.dest_qp_num);
        println!("qp_attr.rq_psn: {}", qp_attr.rq_psn);
        println!("qp_attr.ah_attr.dlid: {}", qp_attr.ah_attr.dlid);
        println!("qp_attr.ah_attr.grh.dgid.gid.global.subnet_prefix: {}", unsafe { gid.global.subnet_prefix });
        println!("qp_attr.ah_attr.grh.dgid.gid.global.interface_id: {}", unsafe { gid.global.interface_id });
        println!("qp_attr.ah_attr.grh.sgid_index: {}", qp_attr.ah_attr.grh.sgid_index);
        return Err(CustomError::new("ibv_modify_qp to rtr ".to_string(), ret).into());
    }

    qp_attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = my_psn;
    qp_attr.max_rd_atomic = 1;
    let qp_attr_mask = 
        ibv_qp_attr_mask::IBV_QP_STATE |
        ibv_qp_attr_mask::IBV_QP_TIMEOUT |
        ibv_qp_attr_mask::IBV_QP_RETRY_CNT |
        ibv_qp_attr_mask::IBV_QP_RNR_RETRY |
        ibv_qp_attr_mask::IBV_QP_SQ_PSN |
        ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;

    let ret = unsafe { ibv_modify_qp(qp, &mut qp_attr, qp_attr_mask.0 as i32) };
    if ret != 0 {
        return Err(CustomError::new("ibv_modify_qp to rts ".to_string(), ret).into());
    }
    Ok(())
}

#[derive(Clone)]
pub struct ProtectionDomain(pub *mut ibv_pd);
unsafe impl Send for ProtectionDomain{}
unsafe impl Sync for ProtectionDomain{}
impl ProtectionDomain{
    pub fn new(context: IbvContext) -> anyhow::Result<ProtectionDomain, CustomError>{
        let pd = unsafe { ibv_alloc_pd(context.ibv_context()) };
        if pd == null_mut() {
            return Err(CustomError::new("ibv_alloc_pd".to_string(), -1).into());
        }
        //let pd = unsafe { Box::from_raw(pd) };
        //let pd = Box::into_raw(pd);
        Ok(ProtectionDomain(pd))
    }
    pub fn pd(&self) -> *mut ibv_pd{
        self.0
    }
}

impl Drop for ProtectionDomain{
    fn drop(&mut self){
        unsafe { ibv_dealloc_pd(self.0) };
    }
}

#[derive(Clone)]
pub struct MemoryRegion(pub *mut ibv_mr);
unsafe impl Send for MemoryRegion{}
unsafe impl Sync for MemoryRegion{}
impl MemoryRegion{
    pub fn mr(&self) -> *mut ibv_mr{
        self.0
    }
}

pub struct IbvSendWr(pub *mut ibv_send_wr);
unsafe impl Send for IbvSendWr{}
unsafe impl Sync for IbvSendWr{}
impl IbvSendWr{
    pub fn send_wr(&self) -> *mut ibv_send_wr{
        self.0
    }
}

#[derive(Clone)]
pub struct CompletionQueue(pub *mut ibv_cq);
unsafe impl Send for CompletionQueue{}
unsafe impl Sync for CompletionQueue{}
impl CompletionQueue{
    pub fn new(context: IbvContext, event_channel: EventChannel) -> anyhow::Result<CompletionQueue, CustomError>{
        let cq = unsafe { ibv_create_cq(context.ibv_context(), 4096, null_mut(), event_channel.event_channel(), 0) };
        if cq.is_null() {
            return Err(CustomError::new("ibv_create_cq".to_string(), -1).into());
        }
        //let cq = unsafe { Box::from_raw(cq) };
        //let cq = Box::into_raw(cq);
        Ok(CompletionQueue(cq))
    }
    pub fn cq(&self) -> *mut ibv_cq{
        self.0
    }
}

impl Drop for CompletionQueue{
    fn drop(&mut self){
        unsafe { ibv_destroy_cq(self.0) };
    }
}

#[derive(Clone)]
pub struct EventChannel(pub *mut ibv_comp_channel);
unsafe impl Send for EventChannel{}
unsafe impl Sync for EventChannel{}
impl EventChannel{
    pub fn new(context: IbvContext) -> anyhow::Result<EventChannel, CustomError>{
        let channel = unsafe { ibv_create_comp_channel(context.ibv_context()) };
        if channel.is_null() {
            return Err(CustomError::new("ibv_create_comp_channel".to_string(), -1).into());
        }
        //let channel = unsafe { Box::from_raw(channel) };
        //let channel = Box::into_raw(channel);
        Ok(EventChannel(channel))
    }
    pub fn event_channel(&self) -> *mut ibv_comp_channel{
        self.0
    }
}

impl Drop for EventChannel{
    fn drop(&mut self){
        unsafe { ibv_destroy_comp_channel(self.0) };
    }
}

#[derive(Clone)]
pub struct QueuePair(pub *mut ibv_qp);
unsafe impl Send for QueuePair{}
unsafe impl Sync for QueuePair{}
impl QueuePair{
    pub fn new(pd: ProtectionDomain, cq: CompletionQueue) -> anyhow::Result<QueuePair, CustomError>{
        
        if pd.pd().is_null() {
            return Err(CustomError::new("ProtectionDomain is null".to_string(), -1).into());
        }
        if cq.cq().is_null() {
            return Err(CustomError::new("CompletionQueue is null".to_string(), -1).into());
        }
        let mut qp_init_attr = ibv_qp_init_attr {
            qp_context: null_mut(),
            send_cq: cq.cq(),
            recv_cq: cq.cq(),
            srq: null_mut(),
            cap: ibv_qp_cap {
                max_send_wr: 4096,
                max_recv_wr: 4096,
                max_send_sge: 15,
                max_recv_sge: 15,
                max_inline_data: 64,
            },
            qp_type: ibv_qp_type::IBV_QPT_RC,
            sq_sig_all: 0,
        };
        println!("Creating QueuePair");
        let qp = unsafe { ibv_create_qp(pd.pd(), &mut qp_init_attr) };
        if qp.is_null() {
            return Err(CustomError::new("ibv_create_qp".to_string(), -1).into());
        }
        println!("QueuePair created");
        let qp = unsafe { Box::from_raw(qp) };
        let qp = Box::into_raw(qp);
        Ok(QueuePair(qp))
    }
    pub fn connect_qp(qp: *mut ibv_qp, gid: ibv_gid, qpn: u32, psn: u32, my_psn: u32, gidx: i32) -> anyhow::Result<(), CustomError> {
        let mut qp_attr = unsafe { std::mem::zeroed::<ibv_qp_attr>() };
        qp_attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
        qp_attr.path_mtu = ibv_mtu::IBV_MTU_4096;
        qp_attr.dest_qp_num = qpn;
        qp_attr.rq_psn = psn;
        qp_attr.max_dest_rd_atomic = 1;
        qp_attr.min_rnr_timer = 12;
    
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = 1;
        qp_attr.ah_attr.dlid = 0;
        qp_attr.ah_attr.grh.dgid = gid;
        qp_attr.ah_attr.is_global = 1;
        qp_attr.ah_attr.grh.sgid_index = gidx as u8;
        qp_attr.ah_attr.grh.hop_limit = 10;
    
        let qp_attr_mask = 
            ibv_qp_attr_mask::IBV_QP_STATE |
            ibv_qp_attr_mask::IBV_QP_AV |
            ibv_qp_attr_mask::IBV_QP_PATH_MTU |
            ibv_qp_attr_mask::IBV_QP_DEST_QPN |
            ibv_qp_attr_mask::IBV_QP_RQ_PSN |
            ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC |
            ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
        let ret = unsafe { ibv_modify_qp(qp, &mut qp_attr, qp_attr_mask.0 as i32) };
        if ret != 0 {
            println!("remote gid: {}", gid_to_ipv6_string(gid).unwrap());
            println!("qp_attr.dest_qp_num: {}", qp_attr.dest_qp_num);
            println!("qp_attr.rq_psn: {}", qp_attr.rq_psn);
            println!("qp_attr.ah_attr.dlid: {}", qp_attr.ah_attr.dlid);
            println!("qp_attr.ah_attr.grh.dgid.gid.global.subnet_prefix: {}", unsafe { gid.global.subnet_prefix });
            println!("qp_attr.ah_attr.grh.dgid.gid.global.interface_id: {}", unsafe { gid.global.interface_id });
            println!("qp_attr.ah_attr.grh.sgid_index: {}", qp_attr.ah_attr.grh.sgid_index);
            return Err(CustomError::new("ibv_modify_qp to rtr ".to_string(), ret).into());
        }
    
        qp_attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
        qp_attr.timeout = 14;
        qp_attr.retry_cnt = 7;
        qp_attr.rnr_retry = 7;
        qp_attr.sq_psn = my_psn;
        qp_attr.max_rd_atomic = 1;
        let qp_attr_mask = 
            ibv_qp_attr_mask::IBV_QP_STATE |
            ibv_qp_attr_mask::IBV_QP_TIMEOUT |
            ibv_qp_attr_mask::IBV_QP_RETRY_CNT |
            ibv_qp_attr_mask::IBV_QP_RNR_RETRY |
            ibv_qp_attr_mask::IBV_QP_SQ_PSN |
            ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;
    
        let ret = unsafe { ibv_modify_qp(qp, &mut qp_attr, qp_attr_mask.0 as i32) };
        if ret != 0 {
            return Err(CustomError::new("ibv_modify_qp to rts ".to_string(), ret).into());
        }
        Ok(())
    }
    pub fn qp(&self) -> *mut ibv_qp{
        self.0
    }
}
impl Drop for QueuePair{
    fn drop(&mut self){
        unsafe { ibv_destroy_qp(self.0) };
    }
}

pub struct Request{
    pub id: u32
}

#[derive(Default, Clone, Debug)]
pub struct MetaData{
    pub remote_address: u64,
    pub rkey: u32,
    pub lkey: u32,
    pub length: u32,
}

impl MetaData{
    pub const LEN: usize = std::mem::size_of::<MetaData>();
    pub fn addr(&self) -> u64{
        self as *const _ as *mut c_void as u64
    }
    pub fn set_remote_address(&mut self, remote_address: u64){
        self.remote_address = remote_address;
    }
}

pub struct MetaDataWrapper(pub *mut MetaData);

unsafe impl Send for MetaDataWrapper{}
unsafe impl Sync for MetaDataWrapper{}


pub fn tcp_server(address: String, port: u16) -> anyhow::Result<(), CustomError> {
    let address = format!("{}:{}", address, port);
    let listener = TcpListener::bind(address.clone()).map_err(|e| CustomError::new(e.to_string(), -1))?;

    println!("Server listening on {}", address);

    for stream in listener.incoming() {
        let mut stream = stream.map_err(|e| CustomError::new(e.to_string(), -1))?;
        let mut buffer = vec![0; 1024]; // Adjust size if necessary
        stream.read(&mut buffer).map_err(|e| CustomError::new(e.to_string(), -1))?;

        let remote_handle: NcclNetSocketHandle = bincode::deserialize(&buffer).unwrap();
        println!("Received RemoteHandle: {:?}", remote_handle);

        // Echo the struct back
        let encoded: Vec<u8> = bincode::serialize(&remote_handle).unwrap();
        stream.write_all(&encoded).map_err(|e| CustomError::new(e.to_string(), -1))?;
    }
    Ok(())
}

pub fn tcp_client(address: String, port: u16, remote_handle: NcclNetSocketHandle) -> anyhow::Result<(), CustomError> {
    let address = format!("{}:{}", address, port);
    let mut stream = TcpStream::connect(address).map_err(|e| CustomError::new(e.to_string(), -1))?;

    let encoded: Vec<u8> = bincode::serialize(&remote_handle).unwrap();
    stream.write_all(&encoded).map_err(|e| CustomError::new(e.to_string(), -1))?;

    let mut buffer = vec![0; 1024]; // Adjust size if necessary
    stream.read(&mut buffer).map_err(|e| CustomError::new(e.to_string(), -1))?;
    Ok(())
}

pub fn subnet_interface_id_to_v6(subnet_id: u64, interface_id: u64) -> Option<Ipv6Addr> {
    let subnet_prefix_bytes = subnet_id.to_be_bytes();
    let interface_id_bytes = interface_id.to_be_bytes();
    let subnet_prefix_bytes = subnet_prefix_bytes.iter().rev().cloned().collect::<Vec<u8>>();
    let interface_id_bytes = interface_id_bytes.iter().rev().cloned().collect::<Vec<u8>>();
    let mut raw = [0u8; 16];
    raw[..8].copy_from_slice(&subnet_prefix_bytes);
    raw[8..].copy_from_slice(&interface_id_bytes);
    gid_to_ipv6_string(ibv_gid{raw})
}

pub enum SocketCommand{
    Connect{
        socket_comm_list: Vec<SocketComm>,
        qp_list: QpList,
    }
}

const BATCH_SIZE: usize = 2000;

pub unsafe fn send_complete(mut cq: *mut ibv_cq, channel: *mut ibv_comp_channel, iterations: usize, opcode_type: ibv_wc_opcode::Type) -> anyhow::Result<i32, CustomError>{
    let mut ret: c_int;
    //let empty_cq = ptr::null::<ibv_cq>() as *mut _;
    let mut context = ptr::null::<c_void>() as *mut _;

    let mut wc_vec: Vec<ibv_wc> = Vec::with_capacity(BATCH_SIZE);
    let wc_ptr = wc_vec.as_mut_ptr();

    let mut total_wc: i32 = 0;

    let nevents = 1;
    let solicited_only = 0;

    loop {
        ret = ibv_poll_cq(cq, BATCH_SIZE as i32, wc_ptr.wrapping_add(total_wc as usize));
        if ret < 0 {
            return Err(CustomError::new("ibv_poll_cq".to_string(), ret).into());
        }
        total_wc += ret;
        if total_wc >= iterations as i32{
            break;
        }
        ret = ibv_req_notify_cq(cq, solicited_only);
        if ret != 0 {
            return Err(CustomError::new("ibv_req_notify_cq".to_string(), ret).into());
        }
        ret = ibv_poll_cq(cq, BATCH_SIZE as i32, wc_ptr.wrapping_add(total_wc as usize));
        if ret < 0 {
            return Err(CustomError::new("ibv_poll_cq".to_string(), ret).into());
        }
        total_wc += ret;
        if total_wc >= iterations as i32{
            break;
        }
        ret = ibv_get_cq_event(channel, &mut cq, &mut context);
        if ret != 0 {
            return Err(CustomError::new("ibv_get_cq_event".to_string(), ret).into());
        }
        //assert!(cq == empty_cq && context as *mut rdma_cm_id == id.0);
        //assert!(cq == empty_cq);
        ibv_ack_cq_events(cq, nevents);
    }
    if ret < 0 {
        return Err(CustomError::new("ibv_poll_cq".to_string(), ret).into());
    }
    for i in 0..total_wc{
        let wc = wc_ptr.wrapping_add(i as usize);
        let status = (*wc).status;
        let opcode = (*wc).opcode;
        if status != ibv_wc_status::IBV_WC_SUCCESS || opcode != opcode_type{
            return Err(CustomError::new(format!("wc status/opcode {}/{} wrong, expected {}/{}", status, opcode, ibv_wc_status::IBV_WC_SUCCESS, opcode_type).to_string(), -1).into());
        }
    }
    Ok(total_wc)
}
const MAX_MESSAGE_SIZE: i32 = 1024 * 1024;

pub fn create_wr(data: *mut c_void, size: c_int,mr: Box<*mut ibv_mr>, qps: i32) -> Vec<*mut ibv_send_wr>{
    let base_size = size / qps as i32;
    let remainder = size % qps as i32;
    let mut offset = 0;
    let remote_mr_start_addr = unsafe { (*(*mr)).addr };
    let mut wr_list = Vec::new();
    for i in 0..qps{
        let mut size = base_size;
        if i < remainder  {
            size += 1; // Distribute the remainder
        }
        let mut last_wr: *mut ibv_send_wr = ptr::null_mut();
        let mut first_wr: *mut ibv_send_wr = ptr::null_mut();
        while size > 0 {
            let message_size = if size > MAX_MESSAGE_SIZE { MAX_MESSAGE_SIZE } else { size };
            let sge = ibv_sge{
                addr: data as u64 + offset as u64,
                length: message_size as u32,
                lkey: 0,
            };
            let sge = Box::new(sge);
            let sge_ptr: *mut ibv_sge = Box::into_raw(sge);
            let mut wr = unsafe { std::mem::zeroed::<ibv_send_wr>() };
            wr.sg_list = sge_ptr;
            wr.num_sge = 1;
            wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = remote_mr_start_addr as u64 + offset as u64;
            wr.wr.rdma.rkey = unsafe { (*(*mr)).rkey };
            if size == message_size {
                wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;
            } else {
                wr.send_flags = 0;
            }
            let wr = Box::new(wr);
            let wr_ptr: *mut ibv_send_wr = Box::into_raw(wr);

            if !last_wr.is_null() {
                unsafe {
                    (*last_wr).next = wr_ptr; // Link the previous WR to the current WR
                }
            } else {
                first_wr = wr_ptr;
            }
            last_wr = wr_ptr;
            size -= message_size;
            offset += message_size as u64;
        }
        wr_list.push(first_wr);
    }
    wr_list

}