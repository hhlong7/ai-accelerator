"""
this is for the kernel/task reprenstation. the kernel can either go to the a cpu or the accelerator
moved by the scheduler. kernel = (type, dim, fpop, bytes)
run in nanosecs
"""

from dataclasses import dataclass

#kernel type
general_mat_mul_kernel = "gemm"     #can run on accelerator or cpu
simple = "elementwise"      #can run on cpu only, bounded by memory

cpu = "CPU"
accelerator = "ACCELERATOR"

@dataclass
class Kernel:
    kernel_id: int
    kernel_type: str
    queue: float
    fpop: int   #floating point operations, the more fpop the more time it takes to run
    moved_bytes: int    #min mem traffic (read + write once), minimum traffic, 
    work_set: int
    dim: tuple = None
    dtype_bytes: int = 2    #bf16 of real accel like google tpu or apple or nvidia tensors
    parallelize: bool = True    #can split the work to mult cores (cpu)
    resource: str = None #cpu or accelerator
    start: float = None
    finish: float = None
    remaining: float = 1.0  #indicate the fraction of work left to do, where 1 = 100% left and 0 is done

    def __post_init__(self):
        if self.fpop < 0 or self.moved_bytes <= 0:
            raise ValueError(f"Inval kernel {self.kernel_id} with fpop: {self.fpop} must be positive and moved_bytes: {self.moved_bytes} must be greater than 0")
        if self.queue < 0:
            raise ValueError(f"Inval kernel {self.kernel_id} because queue time {self.queue} must be >= 0")

    @property
    def fpops_per_byte(self):
        return self.fpop / self.moved_bytes #if high then compute the bound, if low then memory bound

    def can_run_on(self, kind):
        if kind == cpu:
            return True
        if kind == accelerator and self.kernel_type == general_mat_mul_kernel:
            return True
        return False

    @property
    def done(self):
        return self.finish is not None

    @property
    def latency(self):
        if self.finish is None:
            return None
        return self.finish - self.queue     #computin the time from when k was queued to when it finsiehd

    @property
    def wait(self):
        if self.start is None:
            return None
        return self.start - self.queue   #computing the time from when k was queued to when it started running


#general mat mul (gmm): (m x k) @ (k x n) = (m x n)
def make_gmm_kernel(kernel_id, m, k, n, arrival, dtype_bytes=2):
    if min(m, k, n) <= 0:
        raise ValueError(f"Dimensions of the general MatMul kernel: {(m, k, n)} must be positive, id: {kernel_id}")
    fpop = 2 * m * k * n
    a_bytes = m * k * dtype_bytes #in
    b_bytes = k * n * dtype_bytes #weights
    c_bytes = m * n *dtype_bytes #out

    tot = a_bytes + b_bytes + c_bytes
    return Kernel(kernel_id=kernel_id, kernel_type=general_mat_mul_kernel, queue=arrival,fpop=fpop, moved_bytes=tot, work_set=tot, dim=(m, k, n), dtype_bytes=dtype_bytes)


#simple kernels, read n inputs arryas, write 1 output arry
def make_simple_kernel(kernel_id, n_element, arrival, ops_per_element=1, n_in=1, dtype_bytes=2):
    if n_element <= 0:
        raise ValueError(f"The number of elements: {n_element} must be positive, id: {kernel_id}.")
    fpop = n_element * ops_per_element
    total = n_element * dtype_bytes * (n_in + 1)
    return Kernel(kernel_id=kernel_id, kernel_type=simple, queue=arrival, fpop=fpop, moved_bytes=total, work_set=total, dtype_bytes=dtype_bytes)

