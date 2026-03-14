use std::collections::VecDeque;
use std::time::{Duration, Instant};

const WINDOW_DURATION: Duration = Duration::from_secs(2);

/// Tracks episode rendering performance — how many unique episodes are
/// displayed per second, not UI frame rate.
pub(crate) struct PerfTracker {
    timestamps: VecDeque<Instant>,
}

impl PerfTracker {
    pub(crate) fn new() -> Self {
        Self {
            timestamps: VecDeque::new(),
        }
    }

    pub(crate) fn record_display(&mut self) {
        self.timestamps.push_back(Instant::now());
    }

    fn display_fps(&mut self) -> f64 {
        let now = Instant::now();
        let cutoff = now - WINDOW_DURATION;
        while let Some(front) = self.timestamps.front() {
            if *front < cutoff {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }

        if self.timestamps.len() < 2 {
            return 0.0;
        }
        let oldest = *self.timestamps.front().unwrap();
        let span = now.duration_since(oldest).as_secs_f64();
        if span > 0.0 {
            (self.timestamps.len() - 1) as f64 / span
        } else {
            0.0
        }
    }

    pub(crate) fn fps_text(&mut self) -> String {
        format!("{:.1} FPS", self.display_fps())
    }
}
