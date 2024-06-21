#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

// #include <spatial/core/Platform.hpp>
#include "rtm/math.h"
#include "rtm/types.h"
#include "rtm/impl/compiler_utils.h"
#include "rtm/scalarf.h"
#include "rtm/scalard.h"
#include "rtm/version.h"


RTM_IMPL_FILE_PRAGMA_PUSH

namespace rtm
{
	RTM_IMPL_VERSION_NAMESPACE_BEGIN

#if defined(RTM_SSE2_INTRINSICS)

namespace sse2_permute
{
#define SHUFFLEMASK(a0,a1,b2,b3) ( (a0) | ((a1)<<2) | ((b2)<<4) | ((b3)<<6) )
#define SHUFFLEMASK2(a0,a1) ((a0) | ((a1)<<1))

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Double swizzle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Double Swizzle helpers
	// Templated swizzles required for double shuffles when using __m128d, since we have to break it down in to two separate operations.

	
	template<int index0, int index1>
	RTM_FORCE_INLINE __m128d select_vector_swizzle2(const vector4d& vec)
	{ 
		if constexpr ((index0 <= 1) && (index1 <= 1)) {
			// [0,1]:[0,1]
			return _mm_shuffle_pd(vec.xy, vec.xy, SHUFFLEMASK2(index0, index1));
		} 
		else if constexpr ((index0 <= 1) && (index1 >= 2)) {
			// [0,1]:[2,3]
			return _mm_shuffle_pd(vec.xy, vec.zw, SHUFFLEMASK2(index0, index1 - 2));
		}
		else if constexpr ((index0 >= 2) && (index1 <= 1)) {
			// [2,3]:[0,1]
			return _mm_shuffle_pd(vec.zw, vec.xy, SHUFFLEMASK2(index0 - 2, index1));
		}
		else if constexpr ((index0 >= 2) && (index1 >= 2)) {
			// [2,3]:[2,3]
			return _mm_shuffle_pd(vec.zw, vec.zw, SHUFFLEMASK2(index0 - 2, index1 - 2));
		}

		return vec.xy;
	}



	template<> RTM_FORCE_INLINE __m128d select_vector_swizzle2<0, 1>(const vector4d& vec) { return vec.xy; }
	template<> RTM_FORCE_INLINE __m128d select_vector_swizzle2<2, 3>(const vector4d& vec) { return vec.zw; }

#if defined(RTM_SSE4_INTRINSICS)
	// blend can run on more ports than shuffle, so are preferable even if latency is claimed to be the same.
    template<> RTM_FORCE_INLINE __m128d select_vector_swizzle2<0, 3>(const vector4d& vec) { return _mm_blend_pd(vec.xy, vec.zw, SHUFFLEMASK2(0, 1)); }
    template<> RTM_FORCE_INLINE __m128d select_vector_swizzle2<2, 1>(const vector4d& vec) { return _mm_blend_pd(vec.zw, vec.xy, SHUFFLEMASK2(0, 1)); }
#endif // UE_PLATFORM_MATH_USE_SSE4_1


	// Double swizzle wrapper
	template<int index0, int index1, int index2, int Index3>
	RTM_FORCE_INLINE vector4d vector_swizzle_template(const vector4d& vec)
	{
		static_assert(index0 >= 0 && index0 <= 3 && index1 >= 0 && index1 <= 3 && index2 >= 0 && index2 <= 3 && Index3 >= 0 && Index3 <= 3, "Invalid Index");

//#if RTM_AVX_INTRINSICS
//		return SelectVectorSwizzle<index0, index1, index2, index3>(vec);
//#else
		return vector4d{
			select_vector_swizzle2<index0, index1>(vec),
			select_vector_swizzle2<index2, index3>(vec) };
//#endif
	}

	// Specializations
	template<> RTM_FORCE_INLINE vector4d vector_swizzle_template<0, 1, 2, 3>(const vector4d& vec) { return vec; } // Identity

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Double replicate
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <int Index>
	RTM_FORCE_INLINE __m128d vector_replicate_impl2(const __m128d& vec)
	{
		// Note: 2 doubles (VectorRegister2Double / m128d)
		return _mm_shuffle_pd(vec, vec, SHUFFLEMASK2(Index, Index));
	}

	// Double replicate (4 doubles)
	template <int Index, typename std::enable_if< (Index <= 1), bool >::type = true >
	RTM_FORCE_INLINE vector4d vector_replicate_impl4(const vector4d& vec)
	{
		__m128d Temp = vector_replicate_impl2<Index>(vec.xy);
		return vector4d{ Temp, Temp };
	}

	template <int Index, typename std::enable_if< (Index >= 2), bool >::type = true >
	RTM_FORCE_INLINE vector4d vector_replicate_impl4(const vector4d& vec)
	{
		__m128d Temp = vector_replicate_impl2<Index - 2>(vec.zw);
		return vector4d{ Temp, Temp };
	}

	//
	// Double replicate wrapper
	//
	template<int Index>
	RTM_FORCE_INLINE vector4d vector_replicate_template(const vector4d& vec)
	{
		static_assert(Index >= 0 && Index <= 3, "Invalid Index");

#if defined(RTM_AVX2_INTRINSICS)
		return vector_swizzle_template<Index, Index, Index, Index>(vec);
#else
		return vector_replicate_impl4<Index>(vec);
#endif
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Double shuffle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Non-AVX implementation
	template<int index0, int index1, int index2, int index3>
	RTM_FORCE_INLINE vector4d select_vector_shuffle(const vector4d& vec1, const vector4d& vec2)
	{
		return vector4d{
			select_vector_swizzle2<index0, index1>(vec1),
			select_vector_swizzle2<index2, index3>(vec2)
		};
	}


	//
	// Double shuffle wrapper
	//
	template<int index0, int index1, int index2, int index3>
	RTM_FORCE_INLINE vector4d vector_shuffle_template(const vector4d& vec1, const vector4d& vec2)
	{
		static_assert(index0 >= 0 && index0 <= 3 && index1 >= 0 && index1 <= 3 && index2 >= 0 && index2 <= 3 && index3 >= 0 && index3 <= 3, "Invalid Index");
		return select_vector_shuffle<index0, index1, index2, index3>(vec1, vec2);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float swizzle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<int index0, int index1, int index2, int index3>
	RTM_FORCE_INLINE vector4f vector_swizzle_template(const vector4f& vec)
	{
		return _mm_shuffle_ps(vec, vec, SHUFFLEMASK(index0, index1, index2, index3));
	}

	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<0, 1, 2, 3>(const vector4f& vec) { return vec; }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<0, 1, 0, 1>(const vector4f& vec) { return _mm_movelh_ps(vec, vec); }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<2, 3, 2, 3>(const vector4f& vec) { return _mm_movehl_ps(vec, vec); }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<0, 0, 1, 1>(const vector4f& vec) { return _mm_unpacklo_ps(vec, vec); }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<2, 2, 3, 3>(const vector4f& vec) { return _mm_unpackhi_ps(vec, vec); }

#if defined(RTM_SSE4_INTRINSICS)
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<0, 0, 2, 2>(const vector4f& vec) { return _mm_moveldup_ps(vec); }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<1, 1, 3, 3>(const vector4f& vec) { return _mm_movehdup_ps(vec); }
#endif

#if defined(RTM_AVX2_INTRINSICS)
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_template<0, 0, 0, 0>(const vector4f& vec) { return _mm_broadcastss_ps(vec); }
#endif

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float replicate
	template<int Index>
	RTM_FORCE_INLINE vector4f vector_replicate_template(const vector4f& vec)
	{
		static_assert(Index >= 0 && Index <= 3, "Invalid Index");
		return vector_swizzle_template<Index, Index, Index, Index>(vec);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float shuffle
	template<int index0, int index1, int index2, int index3>
	RTM_FORCE_INLINE vector4f vector_shuffle_template(const vector4f& vec1, const vector4f& vec2)
	{
		static_assert(index0 >= 0 && index0 <= 3 && index1 >= 0 && index1 <= 3 && index2 >= 0 && index2 <= 3 && index3 >= 0 && index3 <= 3, "Invalid Index");
		return _mm_shuffle_ps(vec1, vec2, SHUFFLEMASK(index0, index1, index2, index3));
	}

	// Float Shuffle specializations
	template<> RTM_FORCE_INLINE vector4f vector_shuffle_template<0, 1, 0, 1>(const vector4f& vec1, const vector4f& vec2) { return _mm_movelh_ps(vec1, vec2); }
	template<> RTM_FORCE_INLINE vector4f vector_shuffle_template<2, 3, 2, 3>(const vector4f& vec1, const vector4f& vec2) { return _mm_movehl_ps(vec2, vec1); } // Note: movehl copies first from the 2nd argument

}; // namespace sse2_permute


#define VECTOR_REPLICATE( vec, element_index )	sse2_permute::vector_replicate_template<element_index>(vec)
#define VECTOR_SWIZZLE( vec, x, y, z, w )		sse2_permute::vector_swizzle_template<x,y,z,w>( vec )
#define VECTOR_SHUFFLE( vec1, vec2, x, y, z, w )		sse2_permute::vector_shuffle_template<x,y,z,w>( vec1, vec2 )

#elif defined(RTM_NEON_INTRINSICS) && defined(__clang__)
//now we only support __clang__ neon here

template <int x, int y, int z, int w>
RTM_FORCE_INLINE vector4f vector_swizzle_impl(vector4f vec)
{
	return __builtin_shufflevector(vec, vec, x, y, z, w);
}

template <int x, int y, typename std::enable_if < (x <= 1) && (y <= 1), bool >::type = true>
RTM_FORCE_INLINE float64x2_t vector_swizzle_impl2(vector4d vec)
{
	return __builtin_shufflevector(vec.xy, vec.xy, x, y);
}

template <int x, int y, typename std::enable_if < (x <= 1) && (y > 1), bool >::type = true>
RTM_FORCE_INLINE float64x2_t vector_swizzle_impl2(vector4d vec)
{
	return __builtin_shufflevector(vec.xy, vec.zw, x, y);
}

template <int x, int y, typename std::enable_if < (x > 1) && (y <= 1), bool >::type = true>
RTM_FORCE_INLINE float64x2_t vector_swizzle_impl2(vector4d vec)
{
	return __builtin_shufflevector(vec.zw, vec.xy, x - 2, y + 2);
}

template <int x, int y, typename std::enable_if < (x > 1) && (y > 1), bool >::type = true>
RTM_FORCE_INLINE float64x2_t vector_swizzle_impl2(vector4d vec)
{
	return __builtin_shufflevector(vec.zw, vec.zw, x - 2, y);
}

template <int x, int y, int z, int w>
RTM_FORCE_INLINE vector4d vector_swizzle_impl(vector4d vec)
{
	vector4d r;
	r.xy = vector_swizzle_impl2<x, y>(vec);
	r.zw = vector_swizzle_impl2<z, w>(vec);
	return r;
}



template <int x, int y, int z, int w>
RTM_FORCE_INLINE vector4f vector_shuffle_impl(vector4f vec1, vector4f vec2)
{
	return __builtin_shufflevector(vec1, vec2, x, y, z + 4, w + 4);
}

template <int x, int y, int z, int w>
RTM_FORCE_INLINE vector4d vector_shuffle_impl(vector4d vec1, vector4d vec2)
{
	vector4d r;
	r.xy = vector_swizzle_impl2<x, y>(vec1);
	r.zw = vector_swizzle_impl2<z, w>(vec2);
	return r;
}

template <int element_index>
RTM_FORCE_INLINE vector4f vector_replicate_impl(const vector4f& vec)
{
	return vdupq_n_f32(vgetq_lane_f32(vec, element_index));
}

template <int element_index>
RTM_FORCE_INLINE float64x2_t vector_replicate_impl(const float64x2_t& vec)
{
	return vdupq_n_f64(vgetq_lane_f64(vec, element_index));
}

template <int element_index, typename std::enable_if < (element_index <= 1), bool >::type = true >
RTM_FORCE_INLINE vector4d vector_replicate_impl(const vector4d& vec)
{
	vector4d r;
	r.xy = vector_replicate_impl<element_index>(vec.xy);
	r.zw = r.xy;
	return r;
}

template <int element_index, typename std::enable_if < (element_index > 1), bool >::type = true >
RTM_FORCE_INLINE vector4d vector_replicate_impl(const vector4d& vec)
{
	vector4d r;
	r.zw = vector_replicate_impl<element_index - 2>(vec.zw);
	r.xy = r.zw;
	return r;
}

#define VECTOR_REPLICATE( vec, element_index ) vector_replicate_impl<element_index>(vec)
#define VECTOR_SWIZZLE( vec, x, y, z, w ) vector_swizzle_impl<x, y, z, w>(vec)
#define VECTOR_SHUFFLE( vec1, vec2, x, y, z, w )	vector_shuffle_impl<x, y, z, w>(vec1, vec2)

#else
#pragma error("vector swizzle not implement here!");
#endif

RTM_IMPL_VERSION_NAMESPACE_END
}

RTM_IMPL_FILE_PRAGMA_POP

