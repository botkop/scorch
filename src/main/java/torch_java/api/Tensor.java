package torch_java.api;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import torch_java.api.Functions.*;

@Platform(
        include = {
                "/home/nazar/libtorch/include/ATen/ATen.h"
        })

@Namespace("at") @NoOffset public  class Tensor extends Pointer {

    static {
        //System.load("/home/nazar/CLionProjects/torch_app/cmake-build-debug/libtchs.so");
        System.load("/home/nazar/libtorch/lib/libtorch.so");
        System.load("/home/nazar/libtorch/lib/libcaffe2.so");
        // System.loadLibrary("cudart_static");
        System.load("/usr/lib/x86_64-linux-gnu/librt.so");
        System.load("/usr/local/cuda/lib64/libnvToolsExt.so");
        System.load("/usr/local/cuda/lib64/libnvrtc.so");
        System.load("/usr/lib/x86_64-linux-gnu/libcuda.so");
        System.load("/home/nazar/libtorch/lib/libcaffe2_gpu.so");
        System.loadLibrary("dl");

        System.load("/home/nazar/java_torch_2/so/libjava_torch_lib.so");
    }

    public Tensor() {
        super((Pointer)null);
        allocate();
    }

    public Tensor(Tensor other) {
        super(other);
        allocate(other);
    }

    private native void allocate();
    private native void allocate(@ByRef Tensor tensor);

    public native @Cast("int64_t") long dim();

    @Name("operator+=") public native @ByRef Tensor add(@ByRef Tensor other);

    public native @Cast("const char *") String toString();

    public native @Cast("int64_t") long storage_offset();

    public native boolean defined();

    public native void reset();

    public native boolean is_same(@ByRef Tensor tensor);
    public native @Cast("size_t") long use_count();
    public native @Cast("size_t") long weak_use_count();

    public native @ByVal IntList sizes();
    public native @ByVal IntList strides();
    public native @Cast("int64_t") long ndimension();
    public native boolean is_contiguous();
    public native @ByRef Type type();
    //public native TensorTypeId type_id();
    //public native ScalarType scalar_type();
    //public native const Storage& storage();
    public native boolean is_alias_of(@ByRef Tensor tensor);
    public native @ByVal Tensor toType(@ByRef Type t, boolean non_blocking);
    public native @ByRef Tensor copy_(@ByRef Tensor tensor, boolean non_blocking);
    //public native Tensor toType(ScalarType t);
    //public native Tensor toBackend(Backend b);

    public native boolean is_variable();

    /// Returns a `Tensor`'s layout. Defined in Type.h
    //public native Layout layout();

    /// Returns a `Tensor`'s dtype (`TypeMeta`). Defined in TensorMethods.h
    //public native caffe2::TypeMeta dtype();

    /// Returns a `Tensor`'s device.
    //public native Device device();

    /// Returns a `Tensor`'s device index.
    public native @Cast("int64_t") long get_device();

    /// Returns if a `Tensor` has CUDA backend.
    public native boolean is_cuda();

    /// Returns if a `Tensor` has HIP backend.
    public native boolean is_hip();

    /// Returns if a `Tensor` has sparse backend.
    public native boolean is_sparse();

    /// Returns the `TensorOptions` corresponding to this `Tensor`. Defined in
    /// TensorOptions.h.
    public native @ByVal TensorOptions options();

//    template<typename T>
//    T * data() const;
//
//    template <typename T>
//    T item() const;

    // Purposely not defined here to avoid inlining
    public native void print();

    // Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar type and
    // dimension.
//    template<typename T, size_t N>
//    TensorAccessor<T,N> accessor() const& {
//        static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
//        AT_CHECK(dim() == N, "expected ", N, " dims but tensor has ", dim());
//        return TensorAccessor<T,N>(data<T>(),sizes().data(),strides().data());
//    }
//    template<typename T, size_t N>
//    TensorAccessor<T,N> accessor() && = delete;

    // Return a `PackedTensorAccessor` for CUDA `Tensor`s. You have to specify scalar type and
    // dimension. You can optionally specify RestrictPtrTraits as a template parameter to
    // cast the data pointer to a __restrict__ pointer.
    // In order to use this, your CUDA kernel has to take a corresponding PackedTensorAccessor
    // as an argument.
//    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
//    PackedTensorAccessor<T,N,PtrTraits,index_t> packed_accessor() const& {
//        static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
//        AT_CHECK(dim() == N, "expected ", N, " dims but tensor has ", dim());
//        return PackedTensorAccessor<T,N,PtrTraits,index_t>(static_cast<typename PtrTraits<T>::PtrType>(data<T>()),sizes().data(),strides().data());
//    }
//    template<typename T, size_t N,  template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
//    PackedTensorAccessor<T,N> packed_accessor() && = delete;

//    Tensor operator-() const;
//    Tensor& operator+=(const Tensor & other);
//    Tensor& operator+=(Scalar other);
//    Tensor& operator-=(const Tensor & other);
//    Tensor& operator-=(Scalar other);
//    Tensor& operator*=(const Tensor & other);
//    Tensor& operator*=(Scalar other);
//    Tensor& operator/=(const Tensor & other);
//    Tensor& operator/=(Scalar other);
//    Tensor operator[](Scalar index) const;
//    Tensor operator[](Tensor index) const;
//    Tensor operator[](int64_t index) const;
//
//    Tensor cpu() const;
//    Tensor cuda() const;
//    Tensor hip() const;
//
//    // ~~~~~ Autograd API ~~~~~
//
//    Tensor& set_requires_grad(bool requires_grad) {
//        impl_->set_requires_grad(requires_grad);
//        return *this;
//    }
//    bool requires_grad() const {
//        return impl_->requires_grad();
//    }
//
//    Tensor& grad() {
//        return impl_->grad();
//    }
//  const Tensor& grad() const {
//        return impl_->grad();
//    }
//
//    void set_data(Tensor new_data);
//
//    /// Computes the gradient of current tensor w.r.t. graph leaves.
//    void backward(
//            c10::optional<Tensor> gradient = c10::nullopt,
//            bool keep_graph = false,
//            bool create_graph = false);
//
//    // STOP.  Thinking of adding a method here, which only makes use
//    // of other ATen methods?  Define it in native_functions.yaml.
//
//    //example
//    //Tensor * add(Tensor & b);
//    Tensor abs() const;
//    Tensor & abs_();
//    Tensor acos() const;
//    Tensor & acos_();
//    Tensor add(const Tensor & other, Scalar alpha=1) const;
//    Tensor & add_(const Tensor & other, Scalar alpha=1);
//    Tensor add(Scalar other, Scalar alpha=1) const;
//    Tensor & add_(Scalar other, Scalar alpha=1);
//    Tensor addmv(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;
//    Tensor & addmv_(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
//    Tensor addr(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;
//    Tensor & addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
//    Tensor all(int64_t dim, bool keepdim=false) const;
//    bool allclose(const Tensor & other, double rtol=1e-05, double atol=1e-08, bool equal_nan=false) const;
//    Tensor any(int64_t dim, bool keepdim=false) const;
//    Tensor argmax(int64_t dim, bool keepdim=false) const;
//    Tensor argmax() const;
//    Tensor argmin(int64_t dim, bool keepdim=false) const;
//    Tensor argmin() const;
//    Tensor as_strided(IntList size, IntList stride) const;
//    Tensor & as_strided_(IntList size, IntList stride);
//    Tensor as_strided(IntList size, IntList stride, int64_t storage_offset) const;
//    Tensor & as_strided_(IntList size, IntList stride, int64_t storage_offset);
//    Tensor asin() const;
//    Tensor & asin_();
//    Tensor atan() const;
//    Tensor & atan_();
//    Tensor baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
//    Tensor & baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
//    Tensor bernoulli(Generator * generator=nullptr) const;
//    Tensor & bernoulli_(const Tensor & p, Generator * generator=nullptr);
//    Tensor & bernoulli_(double p=0.5, Generator * generator=nullptr);
//    Tensor bernoulli(double p, Generator * generator=nullptr) const;
//    Tensor bincount(const Tensor & weights={}, int64_t minlength=0) const;
//    Tensor bmm(const Tensor & mat2) const;
//    Tensor ceil() const;
//    Tensor & ceil_();
//    std::vector<Tensor> chunk(int64_t chunks, int64_t dim=0) const;
//    Tensor clamp(c10::optional<Scalar> min=c10::nullopt, c10::optional<Scalar> max=c10::nullopt) const;
//    Tensor & clamp_(c10::optional<Scalar> min=c10::nullopt, c10::optional<Scalar> max=c10::nullopt);
//    Tensor clamp_max(Scalar max) const;
//    Tensor & clamp_max_(Scalar max);
//    Tensor clamp_min(Scalar min) const;
//    Tensor & clamp_min_(Scalar min);
//    Tensor contiguous() const;
//    Tensor cos() const;
//    Tensor & cos_();
//    Tensor cosh() const;
//    Tensor & cosh_();
//    Tensor cumsum(int64_t dim, ScalarType dtype) const;
//    Tensor cumsum(int64_t dim) const;
//    Tensor cumprod(int64_t dim, ScalarType dtype) const;
//    Tensor cumprod(int64_t dim) const;
//    Tensor det() const;
//    Tensor diag_embed(int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1) const;
//    Tensor diagflat(int64_t offset=0) const;
//    Tensor diagonal(int64_t offset=0, int64_t dim1=0, int64_t dim2=1) const;
//    Tensor div(const Tensor & other) const;
//    Tensor & div_(const Tensor & other);
//    Tensor div(Scalar other) const;
//    Tensor & div_(Scalar other);
//    Tensor dot(const Tensor & tensor) const;
//    Tensor & resize_(IntList size);
//    Tensor erf() const;
//    Tensor & erf_();
//    Tensor erfc() const;
//    Tensor & erfc_();
//    Tensor exp() const;
//    Tensor & exp_();
//    Tensor expm1() const;
//    Tensor & expm1_();
//    Tensor expand(IntList size, bool implicit=false) const;
//    Tensor expand_as(const Tensor & other) const;
//    Tensor flatten(int64_t start_dim=0, int64_t end_dim=-1) const;
//    Tensor & fill_(Scalar value);
//    Tensor & fill_(const Tensor & value);
//    Tensor floor() const;
//    Tensor & floor_();
//    Tensor ger(const Tensor & vec2) const;
//    std::tuple<Tensor,Tensor> gesv(const Tensor & A) const;
//    Tensor fft(int64_t signal_ndim, bool normalized=false) const;
//    Tensor ifft(int64_t signal_ndim, bool normalized=false) const;
//    Tensor rfft(int64_t signal_ndim, bool normalized=false, bool onesided=true) const;
//    Tensor irfft(int64_t signal_ndim, bool normalized=false, bool onesided=true, IntList signal_sizes={}) const;
//    Tensor index(TensorList indices) const;
//    Tensor & index_copy_(int64_t dim, const Tensor & index, const Tensor & source);
//    Tensor index_put(TensorList indices, const Tensor & values, bool accumulate=false) const;
//    Tensor & index_put_(TensorList indices, const Tensor & values, bool accumulate=false);
//    Tensor inverse() const;
//    Tensor isclose(const Tensor & other, double rtol=1e-05, double atol=1e-08, bool equal_nan=false) const;
//    bool is_distributed() const;
//    bool is_floating_point() const;
//    bool is_complex() const;
//    bool is_nonzero() const;
//    bool is_same_size(const Tensor & other) const;
//    bool is_signed() const;
//    std::tuple<Tensor,Tensor> kthvalue(int64_t k, int64_t dim=-1, bool keepdim=false) const;
//    Tensor log() const;
//    Tensor & log_();
//    Tensor log10() const;
//    Tensor & log10_();
//    Tensor log1p() const;
//    Tensor & log1p_();
//    Tensor log2() const;
//    Tensor & log2_();
//    Tensor logdet() const;
//    Tensor log_softmax(int64_t dim, ScalarType dtype) const;
//    Tensor log_softmax(int64_t dim) const;
//    Tensor logsumexp(int64_t dim, bool keepdim=false) const;
//    Tensor matmul(const Tensor & other) const;
//    Tensor matrix_power(int64_t n) const;
//    std::tuple<Tensor,Tensor> max(int64_t dim, bool keepdim=false) const;
//    Tensor max_values(int64_t dim, bool keepdim=false) const;
//    Tensor mean(ScalarType dtype) const;
//    Tensor mean() const;
//    Tensor mean(IntList dim, bool keepdim, ScalarType dtype) const;
//    Tensor mean(IntList dim, bool keepdim=false) const;
//    Tensor mean(IntList dim, ScalarType dtype) const;
//    std::tuple<Tensor,Tensor> median(int64_t dim, bool keepdim=false) const;
//    std::tuple<Tensor,Tensor> min(int64_t dim, bool keepdim=false) const;
//    Tensor min_values(int64_t dim, bool keepdim=false) const;
//    Tensor mm(const Tensor & mat2) const;
//    std::tuple<Tensor,Tensor> mode(int64_t dim=-1, bool keepdim=false) const;
//    Tensor mul(const Tensor & other) const;
//    Tensor & mul_(const Tensor & other);
//    Tensor mul(Scalar other) const;
//    Tensor & mul_(Scalar other);
//    Tensor mv(const Tensor & vec) const;
//    Tensor mvlgamma(int64_t p) const;
//    Tensor & mvlgamma_(int64_t p);
//    Tensor narrow_copy(int64_t dim, int64_t start, int64_t length) const;
//    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
//    Tensor permute(IntList dims) const;
//    Tensor pin_memory() const;
//    Tensor pinverse(double rcond=1e-15) const;
//    Tensor repeat(IntList repeats) const;
//    Tensor reshape(IntList shape) const;
//    Tensor reshape_as(const Tensor & other) const;
//    Tensor round() const;
//    Tensor & round_();
//    Tensor relu() const;
//    Tensor & relu_();
//    Tensor prelu(const Tensor & weight) const;
//    std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & weight) const;
//    Tensor hardshrink(Scalar lambd=0.5) const;
//    Tensor hardshrink_backward(const Tensor & grad_out, Scalar lambd) const;
//    Tensor rsqrt() const;
//    Tensor & rsqrt_();
//    Tensor select(int64_t dim, int64_t index) const;
//    Tensor sigmoid() const;
//    Tensor & sigmoid_();
//    Tensor sin() const;
//    Tensor & sin_();
//    Tensor sinh() const;
//    Tensor & sinh_();
//    Tensor detach() const;
//    Tensor & detach_();
//    int64_t size(int64_t dim) const;
//    Tensor slice(int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1) const;
//    std::tuple<Tensor,Tensor> slogdet() const;
//    Tensor smm(const Tensor & mat2) const;
//    Tensor softmax(int64_t dim, ScalarType dtype) const;
//    Tensor softmax(int64_t dim) const;
//    std::vector<Tensor> split(int64_t split_size, int64_t dim=0) const;
//    std::vector<Tensor> split_with_sizes(IntList split_sizes, int64_t dim=0) const;
//    Tensor squeeze() const;
//    Tensor squeeze(int64_t dim) const;
//    Tensor & squeeze_();
//    Tensor & squeeze_(int64_t dim);
//    Tensor sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
//    Tensor stft(int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor & window={}, bool normalized=false, bool onesided=true) const;
//    int64_t stride(int64_t dim) const;
//    Tensor sum(ScalarType dtype) const;
//    Tensor sum() const;
//    Tensor sum(IntList dim, bool keepdim, ScalarType dtype) const;
//    Tensor sum(IntList dim, bool keepdim=false) const;
//    Tensor sum(IntList dim, ScalarType dtype) const;
//    Tensor sqrt() const;
//    Tensor & sqrt_();
//    Tensor std(bool unbiased=true) const;
//    Tensor std(int64_t dim, bool unbiased=true, bool keepdim=false) const;
//    Tensor prod(ScalarType dtype) const;
//    Tensor prod() const;
//    Tensor prod(int64_t dim, bool keepdim, ScalarType dtype) const;
//    Tensor prod(int64_t dim, bool keepdim=false) const;
//    Tensor prod(int64_t dim, ScalarType dtype) const;
//    Tensor t() const;
//    Tensor & t_();
//    Tensor tan() const;
//    Tensor & tan_();
//    Tensor tanh() const;
//    Tensor & tanh_();
//    Tensor transpose(int64_t dim0, int64_t dim1) const;
//    Tensor & transpose_(int64_t dim0, int64_t dim1);
//    Tensor flip(IntList dims) const;
//    Tensor roll(IntList shifts, IntList dims={}) const;
//    Tensor rot90(int64_t k=1, IntList dims={0,1}) const;
//    Tensor trunc() const;
//    Tensor & trunc_();
//    Tensor type_as(const Tensor & other) const;
//    Tensor unsqueeze(int64_t dim) const;
//    Tensor & unsqueeze_(int64_t dim);
//    Tensor var(bool unbiased=true) const;
//    Tensor var(int64_t dim, bool unbiased=true, bool keepdim=false) const;
//    Tensor view_as(const Tensor & other) const;
//    Tensor where(const Tensor & condition, const Tensor & other) const;
//    Tensor norm(Scalar p=2) const;
//    Tensor norm(Scalar p, int64_t dim, bool keepdim=false) const;
//    Tensor clone() const;
//    Tensor & resize_as_(const Tensor & the_template);
//    Tensor pow(Scalar exponent) const;
//    Tensor & zero_();
//    Tensor sub(const Tensor & other, Scalar alpha=1) const;
//    Tensor & sub_(const Tensor & other, Scalar alpha=1);
//    Tensor sub(Scalar other, Scalar alpha=1) const;
//    Tensor & sub_(Scalar other, Scalar alpha=1);
//    Tensor addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;
//    Tensor & addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
//    Tensor & sparse_resize_(IntList size, int64_t sparse_dim, int64_t dense_dim);
//    Tensor & sparse_resize_and_clear_(IntList size, int64_t sparse_dim, int64_t dense_dim);
//    Tensor sparse_mask(SparseTensorRef mask) const;
//    Tensor to_dense() const;
//    int64_t sparse_dim() const;
//    int64_t _dimI() const;
//    int64_t dense_dim() const;
//    int64_t _dimV() const;
//    int64_t _nnz() const;
//    Tensor coalesce() const;
//    bool is_coalesced() const;
//    Tensor _indices() const;
//    Tensor _values() const;
//    Tensor & _coalesced_(bool coalesced);
//    Tensor indices() const;
//    Tensor values() const;
//    int64_t numel() const;
//    std::vector<Tensor> unbind(int64_t dim=0) const;
//    Tensor to_sparse(int64_t sparse_dim) const;
//    Tensor to_sparse() const;
//    Tensor to(const TensorOptions & options, bool non_blocking=false, bool copy=false) const;
//    Tensor to(Device device, ScalarType dtype, bool non_blocking=false, bool copy=false) const;
//    Tensor to(ScalarType dtype, bool non_blocking=false, bool copy=false) const;
//    Tensor to(const Tensor & other, bool non_blocking=false, bool copy=false) const;
//    Scalar item() const;
//    void* data_ptr() const;
//    Tensor & set_(Storage source);
//    Tensor & set_(Storage source, int64_t storage_offset, IntList size, IntList stride={});
//    Tensor & set_(const Tensor & source);
//    Tensor & set_();
//    bool is_set_to(const Tensor & tensor) const;
//    Tensor & masked_fill_(const Tensor & mask, Scalar value);
//    Tensor & masked_fill_(const Tensor & mask, const Tensor & value);
//    Tensor & masked_scatter_(const Tensor & mask, const Tensor & source);
//    Tensor view(IntList size) const;
//    Tensor & put_(const Tensor & index, const Tensor & source, bool accumulate=false);
//    Tensor & index_add_(int64_t dim, const Tensor & index, const Tensor & source);
//    Tensor & index_fill_(int64_t dim, const Tensor & index, Scalar value);
//    Tensor & index_fill_(int64_t dim, const Tensor & index, const Tensor & value);
//    Tensor & scatter_(int64_t dim, const Tensor & index, const Tensor & src);
//    Tensor & scatter_(int64_t dim, const Tensor & index, Scalar value);
//    Tensor & scatter_add_(int64_t dim, const Tensor & index, const Tensor & src);
//    Tensor & lt_(Scalar other);
//    Tensor & lt_(const Tensor & other);
//    Tensor & gt_(Scalar other);
//    Tensor & gt_(const Tensor & other);
//    Tensor & le_(Scalar other);
//    Tensor & le_(const Tensor & other);
//    Tensor & ge_(Scalar other);
//    Tensor & ge_(const Tensor & other);
//    Tensor & eq_(Scalar other);
//    Tensor & eq_(const Tensor & other);
//    Tensor & ne_(Scalar other);
//    Tensor & ne_(const Tensor & other);
//    Tensor __and__(Scalar other) const;
//    Tensor __and__(const Tensor & other) const;
//    Tensor & __iand__(Scalar other);
//    Tensor & __iand__(const Tensor & other);
//    Tensor __or__(Scalar other) const;
//    Tensor __or__(const Tensor & other) const;
//    Tensor & __ior__(Scalar other);
//    Tensor & __ior__(const Tensor & other);
//    Tensor __xor__(Scalar other) const;
//    Tensor __xor__(const Tensor & other) const;
//    Tensor & __ixor__(Scalar other);
//    Tensor & __ixor__(const Tensor & other);
//    Tensor __lshift__(Scalar other) const;
//    Tensor __lshift__(const Tensor & other) const;
//    Tensor & __ilshift__(Scalar other);
//    Tensor & __ilshift__(const Tensor & other);
//    Tensor __rshift__(Scalar other) const;
//    Tensor __rshift__(const Tensor & other) const;
//    Tensor & __irshift__(Scalar other);
//    Tensor & __irshift__(const Tensor & other);
//    Tensor & lgamma_();
//    Tensor & atan2_(const Tensor & other);
//    Tensor & tril_(int64_t diagonal=0);
//    Tensor & triu_(int64_t diagonal=0);
//    Tensor & digamma_();
//    Tensor & polygamma_(int64_t n);
//    Tensor & erfinv_();
//    Tensor & frac_();
//    Tensor & renorm_(Scalar p, int64_t dim, Scalar maxnorm);
//    Tensor & reciprocal_();
//    Tensor & neg_();
//    Tensor & pow_(Scalar exponent);
//    Tensor & pow_(const Tensor & exponent);
//    Tensor & lerp_(const Tensor & end, Scalar weight);
//    Tensor & sign_();
//    Tensor & fmod_(Scalar other);
//    Tensor & fmod_(const Tensor & other);
//    Tensor & remainder_(Scalar other);
//    Tensor & remainder_(const Tensor & other);
//    Tensor & addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
//    Tensor addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;
//    Tensor & addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
//    Tensor & addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
//    Tensor & random_(int64_t from, int64_t to, Generator * generator=nullptr);
//    Tensor & random_(int64_t to, Generator * generator=nullptr);
//    Tensor & random_(Generator * generator=nullptr);
//    Tensor & uniform_(double from=0, double to=1, Generator * generator=nullptr);
//    Tensor & normal_(double mean=0, double std=1, Generator * generator=nullptr);
//    Tensor & cauchy_(double median=0, double sigma=1, Generator * generator=nullptr);
//    Tensor & log_normal_(double mean=1, double std=2, Generator * generator=nullptr);
//    Tensor & exponential_(double lambd=1, Generator * generator=nullptr);
//    Tensor & geometric_(double p, Generator * generator=nullptr);
//    Tensor diag(int64_t diagonal=0) const;
//    Tensor cross(const Tensor & other, int64_t dim=-1) const;
//    Tensor triu(int64_t diagonal=0) const;
//    Tensor tril(int64_t diagonal=0) const;
//    Tensor trace() const;
//    Tensor ne(Scalar other) const;
//    Tensor ne(const Tensor & other) const;
//    Tensor eq(Scalar other) const;
//    Tensor eq(const Tensor & other) const;
//    Tensor ge(Scalar other) const;
//    Tensor ge(const Tensor & other) const;
//    Tensor le(Scalar other) const;
//    Tensor le(const Tensor & other) const;
//    Tensor gt(Scalar other) const;
//    Tensor gt(const Tensor & other) const;
//    Tensor lt(Scalar other) const;
//    Tensor lt(const Tensor & other) const;
//    Tensor take(const Tensor & index) const;
//    Tensor index_select(int64_t dim, const Tensor & index) const;
//    Tensor masked_select(const Tensor & mask) const;
//    Tensor nonzero() const;
//    Tensor gather(int64_t dim, const Tensor & index) const;
//    Tensor addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
//    Tensor addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;
//    std::tuple<Tensor,Tensor> gels(const Tensor & A) const;
//    std::tuple<Tensor,Tensor> trtrs(const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false) const;
//    std::tuple<Tensor,Tensor> symeig(bool eigenvectors=false, bool upper=true) const;
//    std::tuple<Tensor,Tensor> eig(bool eigenvectors=false) const;
//    std::tuple<Tensor,Tensor,Tensor> svd(bool some=true, bool compute_uv=true) const;
//    Tensor cholesky(bool upper=false) const;
//    Tensor potrs(const Tensor & input2, bool upper=true) const;
//    Tensor potri(bool upper=true) const;
//    std::tuple<Tensor,Tensor> pstrf(bool upper=true, Scalar tol=-1) const;
//    std::tuple<Tensor,Tensor> qr() const;
//    std::tuple<Tensor,Tensor> geqrf() const;
//    Tensor orgqr(const Tensor & input2) const;
//    Tensor ormqr(const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false) const;
//    std::tuple<Tensor,Tensor> btrifact(bool pivot=true) const;
//    std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(bool pivot=true) const;
//    Tensor btrisolve(const Tensor & LU_data, const Tensor & LU_pivots) const;
//    Tensor multinomial(int64_t num_samples, bool replacement=false, Generator * generator=nullptr) const;
//    Tensor lgamma() const;
//    Tensor digamma() const;
//    Tensor polygamma(int64_t n) const;
//    Tensor erfinv() const;
//    Tensor frac() const;
//    Tensor dist(const Tensor & other, Scalar p=2) const;
//    Tensor reciprocal() const;
//    Tensor neg() const;
//    Tensor atan2(const Tensor & other) const;
//    Tensor lerp(const Tensor & end, Scalar weight) const;
//    Tensor histc(int64_t bins=100, Scalar min=0, Scalar max=0) const;
//    Tensor sign() const;
//    Tensor fmod(Scalar other) const;
//    Tensor fmod(const Tensor & other) const;
//    Tensor remainder(Scalar other) const;
//    Tensor remainder(const Tensor & other) const;
//    Tensor min(const Tensor & other) const;
//    Tensor min() const;
//    Tensor max(const Tensor & other) const;
//    Tensor max() const;
//    Tensor median() const;
//    std::tuple<Tensor,Tensor> sort(int64_t dim=-1, bool descending=false) const;
//    std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true) const;
//    Tensor all() const;
//    Tensor any() const;
//    Tensor renorm(Scalar p, int64_t dim, Scalar maxnorm) const;
//    Tensor unfold(int64_t dimension, int64_t size, int64_t step) const;
//    bool equal(const Tensor & other) const;
//    Tensor pow(const Tensor & exponent) const;
//    Tensor alias() const;
//
//    // We changed .dtype() to return a TypeMeta in #12766. Ideally, we want the
//    // at::kDouble and its friends to be TypeMeta's, but that hasn't happened yet.
//    // Before that change, we make this method to maintain BC for C++ usage like
//    // `x.to(y.dtype)`.
//    // TODO: remove following two after at::kDouble and its friends are TypeMeta's.
//    inline Tensor to(caffe2::TypeMeta type_meta, bool non_blocking=false, bool copy=false) const {
//        return this->to(/*scalar_type=*/typeMetaToScalarType(type_meta), non_blocking, copy);
//    }
//    inline Tensor to(Device device, caffe2::TypeMeta type_meta, bool non_blocking=false, bool copy=false) const {
//        return this->to(device, /*scalar_type=*/typeMetaToScalarType(type_meta), non_blocking, copy);
//    }
//
//    template <typename F, typename... Args>
//    auto m(F func, Args&&... params) const -> decltype(func(*this, std::forward<Args>(params)...)) {
//        return func(*this, std::forward<Args>(params)...);
//    }




    public static void main(String[] args) throws InterruptedException {



        float[] y = {1,2,3,4,5,3,2,3,4,5,4,3,2,2};
        float[] w = {1,2,3,4,5,3,2,3,4,5,4,3,2,2};

        Functions.train(new FloatPointer(y), 100, new FloatPointer(w));

    }
}
