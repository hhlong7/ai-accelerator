import pytest

from sim.kernels import make_gmm_kernel, make_simple_kernel, general_mat_mul_kernel, simple, cpu, accelerator


#gemm 64x128x32 fp16 by hand:
#fpop = 2*64*128*32 = 524288
#A = 64*128*2 = 16384, B = 128*32*2 = 8192, C = 64*32*2 = 4096 => total 28672
def test_gemm_fpop_and_bytes():
    k = make_gmm_kernel(0, 64, 128, 32, arrival=0.00)
    assert k.kernel_type == general_mat_mul_kernel
    assert k.dim == (64, 128, 32)
    assert k.fpop == 524288
    assert k.moved_bytes == 28672
    assert k.fpops_per_byte == pytest.approx(524288 / 28672)


#square gemm n x n x n: fpops_per_byte = 2n^3 / (3 * n^2 * 2 bytes) = n/3
#=> bigger gemm = more compute bound
def test_square_gemm_fpops_per_byte_grows_with_n():
    small = make_gmm_kernel(0, 64, 64, 64, arrival=0.00)
    big = make_gmm_kernel(1, 1024, 1024, 1024, arrival=0.00)
    assert small.fpops_per_byte == pytest.approx(64 / 3)
    assert big.fpops_per_byte == pytest.approx(1024 / 3)


#fp32 doubles the bytes, same fpop
def test_dtype_bytes_scales_traffic():
    half = make_gmm_kernel(0, 64, 128, 32, arrival=0.00, dtype_bytes=2)
    full = make_gmm_kernel(1, 64, 128, 32, arrival=0.00, dtype_bytes=4)
    assert full.fpop == half.fpop
    assert full.moved_bytes == 2 * half.moved_bytes


#simple add of 2 arrays, 1000 elems fp16 by hand:
#fpop = 1000, bytes = 1000 * 2 * (2 inputs + 1 output) = 6000
def test_simple_fpop_and_bytes():
    k = make_simple_kernel(0, 1000, arrival=0.00, ops_per_element=1, n_in=2)
    assert k.kernel_type == simple
    assert k.fpop == 1000
    assert k.moved_bytes == 6000
    assert k.fpops_per_byte < 1.00 #memory bound


def test_can_run_on():
    g = make_gmm_kernel(0, 64, 64, 64, arrival=0.00)
    s = make_simple_kernel(1, 1000, arrival=0.00)
    assert g.can_run_on(cpu) and g.can_run_on(accelerator)
    assert s.can_run_on(cpu) and not s.can_run_on(accelerator)


def test_runtime_fields_start_empty():
    k = make_gmm_kernel(0, 64, 64, 64, arrival=5.00)
    assert k.resource is None
    assert k.remaining == 1.00
    assert not k.done
    assert k.latency is None
    assert k.wait is None


def test_latency_and_wait():
    k = make_gmm_kernel(0, 64, 64, 64, arrival=5.00)
    k.start = 8.00
    k.finish = 20.00
    assert k.done
    assert k.wait == 3.00
    assert k.latency == 15.00


def test_invalid_kernels_raise():
    with pytest.raises(ValueError):
        make_gmm_kernel(0, 0, 64, 64, arrival=0.00)
    with pytest.raises(ValueError):
        make_gmm_kernel(0, 64, 64, 64, arrival=-1.00)
    with pytest.raises(ValueError):
        make_simple_kernel(0, 0, arrival=0.00)