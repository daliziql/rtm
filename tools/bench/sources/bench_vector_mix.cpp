////////////////////////////////////////////////////////////////////////////////
// The MIT License (MIT)
//
// Copyright (c) 2024 Nicholas Frechette & Realtime Math contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////


#include <benchmark/benchmark.h>
#include <rtm/vector4f.h>
#include <rtm/matrix4x4f.h>

using namespace rtm;

namespace {
	template<mix4 comp0, mix4 comp1, mix4 comp2, mix4 comp3>
	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE vector4f RTM_SIMD_CALL vector_mix_optimize(vector4f_arg0 input0, vector4f_arg1 input1) RTM_NO_EXCEPT
	{
#if defined(__clang__)
		constexpr int index0 = (int)comp0;
		constexpr int index1 = (int)comp1;
		constexpr int index2 = (int)comp2;
		constexpr int index3 = (int)comp3;
		return __builtin_shufflevector(input0, input1, index0, index1, index2, index3);
#endif

		return vector_mix<comp0, comp1, comp2, comp3>(input0, input1);
	}

	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f RTM_SIMD_CALL make_tansform(vector4f_arg0 translation, vector4f_arg1 scale, quatf_arg2 orientation) {
		float quatval[4];
		quat_store(orientation, quatval);

		const float x2 = quatval[0] + quatval[0];
		const float y2 = quatval[1] + quatval[1];
		const float z2 = quatval[2] + quatval[2];
		const float xx = quatval[0] * x2;
		const float xy = quatval[0] * y2;
		const float xz = quatval[0] * z2;
		const float yy = quatval[1] * y2;
		const float yz = quatval[1] * z2;
		const float zz = quatval[2] * z2;
		const float wx = quatval[3] * x2;
		const float wy = quatval[3] * y2;
		const float wz = quatval[3] * z2;

		const scalarf scale_x = vector_get_x_as_scalar(scale);
		const scalarf scale_y = vector_get_y_as_scalar(scale);
		const scalarf scale_z = vector_get_z_as_scalar(scale);


		vector4f x_axis = vector_mul(vector_set(1.0F - (yy + zz), xy + wz, xz - wy, 0.0F), scale_x);
		vector4f y_axis = vector_mul(vector_set(xy - wz, 1.0F - (xx + zz), yz + wx, 0.0F), scale_y);
		vector4f z_axis = vector_mul(vector_set(xz + wy, yz - wx, 1.0F - (xx + yy), 0.0F), scale_z);
		vector4f w_axis = translation;

		return matrix4x4f{x_axis, y_axis, z_axis, w_axis};
	}


	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f RTM_SIMD_CALL inverse_normal(matrix4x4f_arg0 input) {
		matrix4x4f output{};

	    vector4f v00 = vector_mix<mix4::x, mix4::x, mix4::y, mix4::y>(input.z_axis, input.z_axis);
	    vector4f v01 = vector_mix<mix4::x, mix4::x, mix4::y, mix4::y>(input.x_axis, input.x_axis);
	    vector4f v02 = vector_mix<mix4::x, mix4::z, mix4::a, mix4::c>(input.z_axis, input.x_axis);
	    vector4f v10 = vector_mix<mix4::z, mix4::w, mix4::z, mix4::w>(input.w_axis, input.w_axis);
	    vector4f v11 = vector_mix<mix4::z, mix4::w, mix4::z, mix4::w>(input.y_axis, input.y_axis);
	    vector4f v12 = vector_mix<mix4::y, mix4::w, mix4::b, mix4::d>(input.w_axis, input.y_axis);

	    vector4f d0 = vector_mul(v00, v10);
	    vector4f d1 = vector_mul(v01, v11);
	    vector4f d2 = vector_mul(v02, v12);

	    v00 = vector_mix<mix4::z, mix4::w, mix4::z, mix4::w>(input.z_axis, input.z_axis);
	    v01 = vector_mix<mix4::z, mix4::w, mix4::z, mix4::w>(input.x_axis, input.x_axis);
	    v02 = vector_mix<mix4::y, mix4::w, mix4::b, mix4::d>(input.z_axis, input.x_axis);
	    v10 = vector_mix<mix4::x, mix4::x, mix4::y, mix4::y>(input.w_axis, input.w_axis);
	    v11 = vector_mix<mix4::x, mix4::x, mix4::y, mix4::y>(input.y_axis, input.y_axis);
	    v12 = vector_mix<mix4::x, mix4::z, mix4::a, mix4::c>(input.w_axis, input.y_axis);

	    d0 = vector_neg_mul_sub(v00, v10, d0);
	    d1 = vector_neg_mul_sub(v01, v11, d1);
	    d2 = vector_neg_mul_sub(v02, v12, d2);

	    v00 = vector_mix<mix4::y, mix4::z, mix4::x, mix4::y>(input.y_axis, input.y_axis);
	    v01 = vector_mix<mix4::z, mix4::x, mix4::y, mix4::x>(input.x_axis, input.x_axis);
	    v02 = vector_mix<mix4::y, mix4::z, mix4::x, mix4::y>(input.w_axis, input.w_axis);
	    vector4f v03 = vector_mix<mix4::z, mix4::x, mix4::y, mix4::x>(input.z_axis, input.z_axis);
	    v10 = vector_mix<mix4::b, mix4::y, mix4::w, mix4::x>(d0, d2);
	    v11 = vector_mix<mix4::w, mix4::b, mix4::y, mix4::z>(d0, d2);
	    v12 = vector_mix<mix4::d, mix4::y, mix4::w, mix4::x>(d1, d2);
	    vector4f v13 = vector_mix<mix4::w, mix4::d, mix4::y, mix4::z>(d1, d2);

	    vector4f c0 = vector_mul(v00, v10);
	    vector4f c2 = vector_mul(v01, v11);
	    vector4f c4 = vector_mul(v02, v12);
	    vector4f c6 = vector_mul(v03, v13);

	    v00 = vector_mix<mix4::z, mix4::w, mix4::y, mix4::z>(input.y_axis, input.y_axis);
	    v01 = vector_mix<mix4::w, mix4::z, mix4::w, mix4::y>(input.x_axis, input.x_axis);
	    v02 = vector_mix<mix4::z, mix4::w, mix4::y, mix4::z>(input.w_axis, input.w_axis);
	    v03 = vector_mix<mix4::w, mix4::z, mix4::w, mix4::y>(input.z_axis, input.z_axis);
	    v10 = vector_mix<mix4::w, mix4::x, mix4::y, mix4::a>(d0, d2);
	    v11 = vector_mix<mix4::z, mix4::y, mix4::a, mix4::x>(d0, d2);
	    v12 = vector_mix<mix4::w, mix4::x, mix4::y, mix4::c>(d1, d2);
	    v13 = vector_mix<mix4::z, mix4::y, mix4::c, mix4::x>(d1, d2);

	    c0 = vector_neg_mul_sub(v00, v10, c0);
	    c2 = vector_neg_mul_sub(v01, v11, c2);
	    c4 = vector_neg_mul_sub(v02, v12, c4);
	    c6 = vector_neg_mul_sub(v03, v13, c6);

	    v00 = vector_mix<mix4::w, mix4::x, mix4::w, mix4::x>(input.y_axis, input.y_axis);
	    v01 = vector_mix<mix4::y, mix4::w, mix4::x, mix4::z>(input.x_axis, input.x_axis);
	    v02 = vector_mix<mix4::w, mix4::x, mix4::w, mix4::x>(input.w_axis, input.w_axis);
	    v03 = vector_mix<mix4::y, mix4::w, mix4::x, mix4::z>(input.z_axis, input.z_axis);
	    v10 = vector_mix<mix4::z, mix4::b, mix4::a, mix4::z>(d0, d2);
	    v11 = vector_mix<mix4::b, mix4::x, mix4::w, mix4::a>(d0, d2);
	    v12 = vector_mix<mix4::z, mix4::d, mix4::c, mix4::z>(d1, d2);
	    v13 = vector_mix<mix4::d, mix4::x, mix4::w, mix4::c>(d1, d2);

	    vector4f c1 = vector_neg_mul_sub(v00, v10, c0);
	    c0 = vector_mul_add(v00, v10, c0);
	    vector4f c3 = vector_mul_add(v01, v11, c2);
	    c2 = vector_neg_mul_sub(v01, v11, c2);
	    vector4f c5 = vector_neg_mul_sub(v02, v12, c4);
	    c4 = vector_mul_add(v02, v12, c4);
	    vector4f c7 = vector_mul_add(v03, v13, c6);
	    c6 = vector_neg_mul_sub(v03, v13, c6);

	    vector4f x_axis = vector_mix<mix4::x, mix4::b, mix4::z, mix4::d>(c0, c1);
	    vector4f y_axis = vector_mix<mix4::x, mix4::b, mix4::z, mix4::d>(c2, c3);
	    vector4f z_axis = vector_mix<mix4::x, mix4::b, mix4::z, mix4::d>(c4, c5);
	    vector4f w_axis = vector_mix<mix4::x, mix4::b, mix4::z, mix4::d>(c6, c7);

	    const scalarf det = vector_dot_as_scalar(x_axis, input.x_axis);
	    const scalarf inv_det_s = scalar_reciprocal(det);
	    const vector4f inv_det = vector_set(inv_det_s);


	    output.x_axis = vector_mul(x_axis, inv_det);
	    output.y_axis = vector_mul(y_axis, inv_det);
	    output.z_axis = vector_mul(z_axis, inv_det);
	    output.w_axis = vector_mul(w_axis, inv_det);

		return output;
	}

	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f RTM_SIMD_CALL inverse_optimize(matrix4x4f_arg0 input) {
		matrix4x4f output{};

	    vector4f v00 = vector_mix_optimize<mix4::x, mix4::x, mix4::y, mix4::y>(input.z_axis, input.z_axis);
	    vector4f v01 = vector_mix_optimize<mix4::x, mix4::x, mix4::y, mix4::y>(input.x_axis, input.x_axis);
	    vector4f v02 = vector_mix_optimize<mix4::x, mix4::z, mix4::a, mix4::c>(input.z_axis, input.x_axis);
	    vector4f v10 = vector_mix_optimize<mix4::z, mix4::w, mix4::z, mix4::w>(input.w_axis, input.w_axis);
	    vector4f v11 = vector_mix_optimize<mix4::z, mix4::w, mix4::z, mix4::w>(input.y_axis, input.y_axis);
	    vector4f v12 = vector_mix_optimize<mix4::y, mix4::w, mix4::b, mix4::d>(input.w_axis, input.y_axis);

	    vector4f d0 = vector_mul(v00, v10);
	    vector4f d1 = vector_mul(v01, v11);
	    vector4f d2 = vector_mul(v02, v12);

	    v00 = vector_mix_optimize<mix4::z, mix4::w, mix4::z, mix4::w>(input.z_axis, input.z_axis);
	    v01 = vector_mix_optimize<mix4::z, mix4::w, mix4::z, mix4::w>(input.x_axis, input.x_axis);
	    v02 = vector_mix_optimize<mix4::y, mix4::w, mix4::b, mix4::d>(input.z_axis, input.x_axis);
	    v10 = vector_mix_optimize<mix4::x, mix4::x, mix4::y, mix4::y>(input.w_axis, input.w_axis);
	    v11 = vector_mix_optimize<mix4::x, mix4::x, mix4::y, mix4::y>(input.y_axis, input.y_axis);
	    v12 = vector_mix_optimize<mix4::x, mix4::z, mix4::a, mix4::c>(input.w_axis, input.y_axis);

	    d0 = vector_neg_mul_sub(v00, v10, d0);
	    d1 = vector_neg_mul_sub(v01, v11, d1);
	    d2 = vector_neg_mul_sub(v02, v12, d2);

	    v00 = vector_mix_optimize<mix4::y, mix4::z, mix4::x, mix4::y>(input.y_axis, input.y_axis);
	    v01 = vector_mix_optimize<mix4::z, mix4::x, mix4::y, mix4::x>(input.x_axis, input.x_axis);
	    v02 = vector_mix_optimize<mix4::y, mix4::z, mix4::x, mix4::y>(input.w_axis, input.w_axis);
	    vector4f v03 = vector_mix_optimize<mix4::z, mix4::x, mix4::y, mix4::x>(input.z_axis, input.z_axis);
	    v10 = vector_mix_optimize<mix4::b, mix4::y, mix4::w, mix4::x>(d0, d2);
	    v11 = vector_mix_optimize<mix4::w, mix4::b, mix4::y, mix4::z>(d0, d2);
	    v12 = vector_mix_optimize<mix4::d, mix4::y, mix4::w, mix4::x>(d1, d2);
	    vector4f v13 = vector_mix_optimize<mix4::w, mix4::d, mix4::y, mix4::z>(d1, d2);

	    vector4f c0 = vector_mul(v00, v10);
	    vector4f c2 = vector_mul(v01, v11);
	    vector4f c4 = vector_mul(v02, v12);
	    vector4f c6 = vector_mul(v03, v13);

	    v00 = vector_mix_optimize<mix4::z, mix4::w, mix4::y, mix4::z>(input.y_axis, input.y_axis);
	    v01 = vector_mix_optimize<mix4::w, mix4::z, mix4::w, mix4::y>(input.x_axis, input.x_axis);
	    v02 = vector_mix_optimize<mix4::z, mix4::w, mix4::y, mix4::z>(input.w_axis, input.w_axis);
	    v03 = vector_mix_optimize<mix4::w, mix4::z, mix4::w, mix4::y>(input.z_axis, input.z_axis);
	    v10 = vector_mix_optimize<mix4::w, mix4::x, mix4::y, mix4::a>(d0, d2);
	    v11 = vector_mix_optimize<mix4::z, mix4::y, mix4::a, mix4::x>(d0, d2);
	    v12 = vector_mix_optimize<mix4::w, mix4::x, mix4::y, mix4::c>(d1, d2);
	    v13 = vector_mix_optimize<mix4::z, mix4::y, mix4::c, mix4::x>(d1, d2);

	    c0 = vector_neg_mul_sub(v00, v10, c0);
	    c2 = vector_neg_mul_sub(v01, v11, c2);
	    c4 = vector_neg_mul_sub(v02, v12, c4);
	    c6 = vector_neg_mul_sub(v03, v13, c6);

	    v00 = vector_mix_optimize<mix4::w, mix4::x, mix4::w, mix4::x>(input.y_axis, input.y_axis);
	    v01 = vector_mix_optimize<mix4::y, mix4::w, mix4::x, mix4::z>(input.x_axis, input.x_axis);
	    v02 = vector_mix_optimize<mix4::w, mix4::x, mix4::w, mix4::x>(input.w_axis, input.w_axis);
	    v03 = vector_mix_optimize<mix4::y, mix4::w, mix4::x, mix4::z>(input.z_axis, input.z_axis);
	    v10 = vector_mix_optimize<mix4::z, mix4::b, mix4::a, mix4::z>(d0, d2);
	    v11 = vector_mix_optimize<mix4::b, mix4::x, mix4::w, mix4::a>(d0, d2);
	    v12 = vector_mix_optimize<mix4::z, mix4::d, mix4::c, mix4::z>(d1, d2);
	    v13 = vector_mix_optimize<mix4::d, mix4::x, mix4::w, mix4::c>(d1, d2);

	    vector4f c1 = vector_neg_mul_sub(v00, v10, c0);
	    c0 = vector_mul_add(v00, v10, c0);
	    vector4f c3 = vector_mul_add(v01, v11, c2);
	    c2 = vector_neg_mul_sub(v01, v11, c2);
	    vector4f c5 = vector_neg_mul_sub(v02, v12, c4);
	    c4 = vector_mul_add(v02, v12, c4);
	    vector4f c7 = vector_mul_add(v03, v13, c6);
	    c6 = vector_neg_mul_sub(v03, v13, c6);

	    vector4f x_axis = vector_mix_optimize<mix4::x, mix4::b, mix4::z, mix4::d>(c0, c1);
	    vector4f y_axis = vector_mix_optimize<mix4::x, mix4::b, mix4::z, mix4::d>(c2, c3);
	    vector4f z_axis = vector_mix_optimize<mix4::x, mix4::b, mix4::z, mix4::d>(c4, c5);
	    vector4f w_axis = vector_mix_optimize<mix4::x, mix4::b, mix4::z, mix4::d>(c6, c7);

	    const scalarf det = vector_dot_as_scalar(x_axis, input.x_axis);
	    const scalarf inv_det_s = scalar_reciprocal(det);
	    const vector4f inv_det = vector_set(inv_det_s);


	    output.x_axis = vector_mul(x_axis, inv_det);
	    output.y_axis = vector_mul(y_axis, inv_det);
	    output.z_axis = vector_mul(z_axis, inv_det);
	    output.w_axis = vector_mul(w_axis, inv_det);

		return output;
	}
}

namespace {
	vector4f g_translation1 = vector_set(4.1f, 4.2f, 4.3f, 1.0f);
	vector4f g_scale = vector_set(1.0f, 1.0f, 1.0f, 1.0f);
	vector4f g_orientation = vector_set(5.1f, 5.2f, 5.3f, 1.0f);
}

class matrix4x4f_agent {
public:
	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f_agent() = default;
	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f_agent(vector4f_arg0 translation, vector4f_arg1 scale, quatf_arg2 orientation) {
		m_value = make_tansform(translation, scale, orientation);
	}

	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f& RTM_SIMD_CALL ref() {
		return m_value;
	}

	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f& RTM_SIMD_CALL ref() const {
		return const_cast<matrix4x4f_agent*>(this)->ref();
	}

	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f_agent RTM_SIMD_CALL inverse_this_normal() const {
		matrix4x4f_agent result{};
		result.ref() = inverse_normal(m_value);
		return result;
	}

	RTM_DISABLE_SECURITY_COOKIE_CHECK RTM_FORCE_INLINE matrix4x4f_agent RTM_SIMD_CALL inverse_this_optimize() const {
		matrix4x4f_agent result{};
		result.ref() = inverse_optimize(m_value);
		return result;
	}

private:
	matrix4x4f m_value{};
};

static void bm_vector_mix_normal(benchmark::State& state) {
	auto vec = std::vector<matrix4x4f_agent>{};
	vec.resize(1000000);

	int index = 0;
	for (auto _ : state) {
		vec[index++ % vec.size()] = matrix4x4f_agent(g_translation1, g_scale, g_orientation).inverse_this_normal();
	}

	benchmark::DoNotOptimize(vec);
}

BENCHMARK(bm_vector_mix_normal);

static void bm_vector_mix_optimize(benchmark::State& state) {
	auto vec = std::vector<matrix4x4f_agent>{};
	vec.resize(1000000);

	int index = 0;
	for (auto _ : state) {
		vec[index++ % vec.size()] = matrix4x4f_agent(g_translation1, g_scale, g_orientation).inverse_this_optimize();
	}

	benchmark::DoNotOptimize(vec);
}

BENCHMARK(bm_vector_mix_optimize);