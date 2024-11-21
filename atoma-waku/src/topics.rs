use waku_bindings::{Encoding, WakuContentTopic};

pub static ATOMA_UPDATE_PUBLIC_ADDRESS: WakuContentTopic =
    WakuContentTopic::new("atoma", "1", "registration", Encoding::Proto);
