import nki.isa as nisa
import nki.language as nl
def kernel_assert(condition, error_text):
    if not condition:
        raise AssertionError(error_text)
def div_ceil(n, d):
    return (n + d - 1) // d
def stream_shuffle_broadcast(src, dst):
    dst_npar = dst.shape[0]
    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(src=src[0:1, :], dst=dst[i * 32 : i * 32 + cur, 0 : dst.shape[1]], shuffle_mask=shuffle_mask)
def ve_transpose_broadcast(src, dst, num_free, num_par):
    _CHUNK = 32
    for c in range(0, num_free, _CHUNK):
        ce = min(c + _CHUNK, num_free)
        nisa.nc_transpose(dst[0:1, c:ce], src[c:ce], engine=nisa.vector_engine)
    stream_shuffle_broadcast(src=dst[0:1, :num_free], dst=dst[:num_par, :num_free])
