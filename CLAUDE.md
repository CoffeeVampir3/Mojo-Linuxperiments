Avoid leading underscores in names (_blah should be blah -- this includes members, in general, just never use leading underscores.)

This project emphasies NUMA awareness. The principle here is that the data should live closest to the most-frequent operation. If there's a frequent-reader but infrequent-writer, we should localize the data to the reader. This can often be done with left-right or dekker patterns in places one might tend to reach for atomics. Even at the large scale for machine learning, we should respect that remote-reads are expensive but manageable. Remote writes are very expensive and should be avoided if possible, prefer remote-reads over remote-writes, and prefer neither if possible.

Avoid comments that add no value. Avoid commenting on unfinished code or prototypes to avoid comments going stale. Avoid comments that are negations ("Doesn't do") -- it's not useful.

This is a mojo project, mojo is very good at simd. Where possible prefer simd over scalar code.

NumaArena is designed to regionalize local memory to a numa domain. BurstPools are designed to run numa-local workers at HFT frequency dispatch/join cycles. The project has a high focus on these concepts and they should be respected.
