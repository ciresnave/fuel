// VecOps trait and all type impls are provided by candle-core-types.
pub use candle_core_types::cpu::kernels::*;

#[inline(always)]
pub fn par_for_each(n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        func(0)
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| func(thread_idx));
            }
        })
    }
}

#[inline(always)]
pub fn par_range(lo: usize, up: usize, n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        for i in lo..up {
            func(i)
        }
    } else {
        let range_len = up - lo;
        let chunk_size = range_len.div_ceil(n_threads);
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                let start = lo + thread_idx * chunk_size;
                let end = (start + chunk_size).min(up);
                if start < up {
                    s.spawn(move |_| {
                        for i in start..end {
                            func(i)
                        }
                    });
                }
            }
        })
    }
}
