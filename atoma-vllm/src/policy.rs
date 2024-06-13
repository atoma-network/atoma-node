use std::{
    collections::VecDeque,
    fmt::Debug,
    time::{Duration, Instant},
};

use crate::sequence::SequenceGroup;

pub trait Policy: Debug {
    fn get_priority(now: Instant, sequence_group: &SequenceGroup) -> Duration;
    fn sort_by_priority(
        now: Instant,
        sequence_groups: &VecDeque<SequenceGroup>,
    ) -> VecDeque<SequenceGroup> {
        let mut output: Vec<SequenceGroup> = sequence_groups.iter().cloned().collect::<Vec<_>>();
        output.sort_by(|v1, v2| {
            Self::get_priority(now, v2)
                .partial_cmp(&Self::get_priority(now, v1))
                .unwrap() // DON'T PANIC: `Duration` admits a complete ordering
        });
        output.into()
    }
}
/// `Policy` - Responsible for deciding which `Sequence`'s to be processed next, on the `Scheduler`
#[derive(Debug)]
pub struct FcfsPolicy {}

impl Policy for FcfsPolicy {
    fn get_priority(now: Instant, sequence_group: &SequenceGroup) -> Duration {
        now - sequence_group.arrival_time()
    }
}
