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

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float swizzle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<int index0, int index1, int index2, int index3>
	FORCEINLINE vector4f vector_swizzle_template(const vector4f& vec)
	{
		return _mm_shuffle_ps(vec, vec, SHUFFLEMASK(index0, index1, index2, index3));
	}

	template<> FORCEINLINE vector4f vector_swizzle_template<0, 1, 2, 3>(const vector4f& vec) { return vec; }
	template<> FORCEINLINE vector4f vector_swizzle_template<0, 1, 0, 1>(const vector4f& vec) { return _mm_movelh_ps(vec, vec); }
	template<> FORCEINLINE vector4f vector_swizzle_template<2, 3, 2, 3>(const vector4f& vec) { return _mm_movehl_ps(vec, vec); }
	template<> FORCEINLINE vector4f vector_swizzle_template<0, 0, 1, 1>(const vector4f& vec) { return _mm_unpacklo_ps(vec, vec); }
	template<> FORCEINLINE vector4f vector_swizzle_template<2, 2, 3, 3>(const vector4f& vec) { return _mm_unpackhi_ps(vec, vec); }

#if defined(RTM_SSE4_INTRINSICS)
	template<> FORCEINLINE vector4f vector_swizzle_template<0, 0, 2, 2>(const vector4f& vec) { return _mm_moveldup_ps(vec); }
	template<> FORCEINLINE vector4f vector_swizzle_template<1, 1, 3, 3>(const vector4f& vec) { return _mm_movehdup_ps(vec); }
#endif

#if defined(RTM_AVX2_INTRINSICS)
	template<> FORCEINLINE vector4f vector_swizzle_template<0, 0, 0, 0>(const vector4f& vec) { return _mm_broadcastss_ps(vec); }
#endif

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float replicate
	template<int Index>
	FORCEINLINE vector4f vector_replicate_template(const vector4f& vec)
	{
		static_assert(Index >= 0 && Index <= 3, "Invalid Index");
		return vector_swizzle_template<Index, Index, Index, Index>(vec);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float shuffle
	template<int index0, int index1, int index2, int index3>
	FORCEINLINE vector4f vector_shuffle_template(const vector4f& vec1, const vector4f& vec2)
	{
		static_assert(index0 >= 0 && index0 <= 3 && index1 >= 0 && index1 <= 3 && index2 >= 0 && index2 <= 3 && index3 >= 0 && index3 <= 3, "Invalid Index");
		return _mm_shuffle_ps(vec1, vec2, SHUFFLEMASK(index0, index1, index2, index3));
	}

	// Float Shuffle specializations
	template<> FORCEINLINE vector4f vector_shuffle_template<0, 1, 0, 1>(const vector4f& vec1, const vector4f& vec2) { return _mm_movelh_ps(vec1, vec2); }
	template<> FORCEINLINE vector4f vector_shuffle_template<2, 3, 2, 3>(const vector4f& vec1, const vector4f& vec2) { return _mm_movehl_ps(vec2, vec1); } // Note: movehl copies first from the 2nd argument

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

template <int x, int y, int z, int w>
RTM_FORCE_INLINE vector4f vector_shuffle_impl(vector4f vec1, vector4f vec2)
{
	return __builtin_shufflevector(vec1, vec2, x, y, z + 4, w + 4);
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

#define VECTOR_REPLICATE( vec, element_index ) vector_replicate_impl<element_index>(vec)
#define VECTOR_SWIZZLE( vec, x, y, z, w ) vector_swizzle_impl<x, y, z, w>(vec)
#define VECTOR_SHUFFLE( vec1, vec2, x, y, z, w )	vector_shuffle_impl<x, y, z, w>(vec1, vec2)

#else
#pragma error("vector swizzle not implement here!");
#endif

RTM_IMPL_VERSION_NAMESPACE_END
}

RTM_IMPL_FILE_PRAGMA_POP

