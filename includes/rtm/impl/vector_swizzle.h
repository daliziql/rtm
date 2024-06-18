#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <spatial/core/Platform.hpp>
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
	// These need to be #define and not constexpr to make them work with enable_if
#define InLane0(Index0, Index1)		((Index0) <= 1 && (Index1) <= 1)
#define InLane1(Index0, Index1)		((Index0) >= 2 && (Index1) >= 2)
#define InSameLane(Index0, Index1)	(InLane0(Index0, Index1) || InLane1(Index0, Index1))
#define OutOfLane(Index0, Index1)	(!InSameLane(Index0, Index1))

#define SHUFFLEMASK(A0,A1,B2,B3) ( (A0) | ((A1)<<2) | ((B2)<<4) | ((B3)<<6) )
#define SHUFFLEMASK2(A0,A1) ((A0) | ((A1)<<1))

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Double swizzle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Double Swizzle helpers
	// Templated swizzles required for double shuffles when using __m128d, since we have to break it down in to two separate operations.

	
	template<int Index0, int Index1>
	FORCEINLINE __m128d SelectVectorSwizzle2(const vector4d& vec)
	{ 
		if constexpr ((Index0 <= 1) && (Index1 <= 1)) {
			// [0,1]:[0,1]
			return _mm_shuffle_pd(vec.xy, vec.xy, SHUFFLEMASK2(Index0, Index1));
		} 
		else if constexpr ((Index0 <= 1) && (Index1 >= 2)) {
			// [0,1]:[2,3]
			return _mm_shuffle_pd(vec.xy, vec.zw, SHUFFLEMASK2(Index0, Index1 - 2));
		}
		else if constexpr ((Index0 >= 2) && (Index1 <= 1)) {
			// [2,3]:[0,1]
			return _mm_shuffle_pd(vec.zw, vec.xy, SHUFFLEMASK2(Index0 - 2, Index1));
		}
		else if constexpr ((Index0 >= 2) && (Index1 >= 2)) {
			// [2,3]:[2,3]
			return _mm_shuffle_pd(vec.zw, vec.zw, SHUFFLEMASK2(Index0 - 2, Index1 - 2));
		}

		return vec.xy;
	}



	template<> FORCEINLINE __m128d SelectVectorSwizzle2<0, 1>(const vector4d& vec) { return vec.xy; }
	template<> FORCEINLINE __m128d SelectVectorSwizzle2<2, 3>(const vector4d& vec) { return vec.zw; }

#if defined(RTM_SSE4_INTRINSICS)
	// blend can run on more ports than shuffle, so are preferable even if latency is claimed to be the same.
    template<> FORCEINLINE __m128d SelectVectorSwizzle2<0, 3>(const vector4d& vec) { return _mm_blend_pd(vec.xy, vec.zw, SHUFFLEMASK2(0, 1)); }
    template<> FORCEINLINE __m128d SelectVectorSwizzle2<2, 1>(const vector4d& vec) { return _mm_blend_pd(vec.zw, vec.xy, SHUFFLEMASK2(0, 1)); }
#endif // UE_PLATFORM_MATH_USE_SSE4_1


	// Double swizzle wrapper
	template<int Index0, int Index1, int Index2, int Index3>
	FORCEINLINE vector4d VectorSwizzleTemplate(const vector4d& vec)
	{
		static_assert(Index0 >= 0 && Index0 <= 3 && Index1 >= 0 && Index1 <= 3 && Index2 >= 0 && Index2 <= 3 && Index3 >= 0 && Index3 <= 3, "Invalid Index");

//#if RTM_AVX_INTRINSICS
//		return SelectVectorSwizzle<Index0, Index1, Index2, Index3>(Vec);
//#else
		return vector4d{
			SelectVectorSwizzle2<Index0, Index1>(vec),
			SelectVectorSwizzle2<Index2, Index3>(vec) };
//#endif
	}

	// Specializations
	template<> FORCEINLINE vector4d VectorSwizzleTemplate<0, 1, 2, 3>(const vector4d& vec) { return vec; } // Identity

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Double replicate
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <int Index>
	FORCEINLINE __m128d VectorReplicateImpl2(const __m128d& vec)
	{
		// Note: 2 doubles (VectorRegister2Double / m128d)
		return _mm_shuffle_pd(vec, vec, SHUFFLEMASK2(Index, Index));
	}

	// Double replicate (4 doubles)
	template <int Index, typename std::enable_if< (Index <= 1), bool >::type = true >
	FORCEINLINE vector4d VectorReplicateImpl4(const vector4d& vec)
	{
		__m128d Temp = VectorReplicateImpl2<Index>(vec.xy);
		return vector4d{ Temp, Temp };
	}

	template <int Index, typename std::enable_if< (Index >= 2), bool >::type = true >
	FORCEINLINE vector4d VectorReplicateImpl4(const vector4d& Vec)
	{
		__m128d Temp = VectorReplicateImpl2<Index - 2>(Vec.zw);
		return vector4d{ Temp, Temp };
	}

	//
	// Double replicate wrapper
	//
	template<int Index>
	FORCEINLINE vector4d VectorReplicateTemplate(const vector4d& Vec)
	{
		static_assert(Index >= 0 && Index <= 3, "Invalid Index");

#if defined(RTM_AVX2_INTRINSICS)
		return VectorSwizzleTemplate<Index, Index, Index, Index>(Vec);
#else
		return VectorReplicateImpl4<Index>(Vec);
#endif
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Double shuffle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Non-AVX implementation
	template<int Index0, int Index1, int Index2, int Index3>
	FORCEINLINE vector4d SelectVectorShuffle(const vector4d& vec1, const vector4d& vec2)
	{
		return vector4d{
			SelectVectorSwizzle2<Index0, Index1>(vec1),
			SelectVectorSwizzle2<Index2, Index3>(vec2)
		};
	}


	//
	// Double shuffle wrapper
	//
	template<int Index0, int Index1, int Index2, int Index3>
	FORCEINLINE vector4d VectorShuffleTemplate(const vector4d& Vec1, const vector4d& Vec2)
	{
		static_assert(Index0 >= 0 && Index0 <= 3 && Index1 >= 0 && Index1 <= 3 && Index2 >= 0 && Index2 <= 3 && Index3 >= 0 && Index3 <= 3, "Invalid Index");
		return SelectVectorShuffle<Index0, Index1, Index2, Index3>(Vec1, Vec2);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float swizzle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<int Index0, int Index1, int Index2, int Index3>
	FORCEINLINE vector4f VectorSwizzleTemplate(const vector4f& Vec)
	{
		return _mm_shuffle_ps(Vec, Vec, SHUFFLEMASK(Index0, Index1, Index2, Index3));
	}

	template<> FORCEINLINE vector4f VectorSwizzleTemplate<0, 1, 2, 3>(const vector4f& Vec) { return Vec; }
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<0, 1, 0, 1>(const vector4f& Vec) { return _mm_movelh_ps(Vec, Vec); }
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<2, 3, 2, 3>(const vector4f& Vec) { return _mm_movehl_ps(Vec, Vec); }
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<0, 0, 1, 1>(const vector4f& Vec) { return _mm_unpacklo_ps(Vec, Vec); }
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<2, 2, 3, 3>(const vector4f& Vec) { return _mm_unpackhi_ps(Vec, Vec); }

#if defined(RTM_SSE4_INTRINSICS)
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<0, 0, 2, 2>(const vector4f& Vec) { return _mm_moveldup_ps(Vec); }
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<1, 1, 3, 3>(const vector4f& Vec) { return _mm_movehdup_ps(Vec); }
#endif

#if defined(RTM_AVX2_INTRINSICS)
	template<> FORCEINLINE vector4f VectorSwizzleTemplate<0, 0, 0, 0>(const vector4f& Vec) { return _mm_broadcastss_ps(Vec); }
#endif

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float replicate
	template<int Index>
	FORCEINLINE vector4f VectorReplicateTemplate(const vector4f& Vec)
	{
		static_assert(Index >= 0 && Index <= 3, "Invalid Index");
		return VectorSwizzleTemplate<Index, Index, Index, Index>(Vec);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float shuffle
	template<int Index0, int Index1, int Index2, int Index3>
	FORCEINLINE vector4f VectorShuffleTemplate(const vector4f& Vec1, const vector4f& Vec2)
	{
		static_assert(Index0 >= 0 && Index0 <= 3 && Index1 >= 0 && Index1 <= 3 && Index2 >= 0 && Index2 <= 3 && Index3 >= 0 && Index3 <= 3, "Invalid Index");
		return _mm_shuffle_ps(Vec1, Vec2, SHUFFLEMASK(Index0, Index1, Index2, Index3));
	}

	// Float Shuffle specializations
	template<> FORCEINLINE vector4f VectorShuffleTemplate<0, 1, 0, 1>(const vector4f& Vec1, const vector4f& Vec2) { return _mm_movelh_ps(Vec1, Vec2); }
	template<> FORCEINLINE vector4f VectorShuffleTemplate<2, 3, 2, 3>(const vector4f& Vec1, const vector4f& Vec2) { return _mm_movehl_ps(Vec2, Vec1); } // Note: movehl copies first from the 2nd argument

}; // namespace sse2_permute


#define VectorReplicate( Vec, ElementIndex )	sse2_permute::VectorReplicateTemplate<ElementIndex>(Vec)

#define VectorSwizzle( Vec, X, Y, Z, W )		sse2_permute::VectorSwizzleTemplate<X,Y,Z,W>( Vec )

#define VectorSwizzle2(Vec, X, Y)				sse2_permute::VectorSwizzleTemplate2<X, Y>(Vec)

#define VectorShuffle( Vec1, Vec2, X, Y, Z, W )		sse2_permute::VectorShuffleTemplate<X,Y,Z,W>( Vec1, Vec2 )

#elif defined(RTM_NEON_INTRINSICS) && defined(__clang__)
//now we only support __clang__ neon here

template <int X, int Y, int Z, int W>
FORCEINLINE vector4f VectorSwizzleImpl(vector4f Vec)
{
	return __builtin_shufflevector(Vec, Vec, X, Y, Z, W);
}

template <int X, int Y, typename std::enable_if < (X <= 1) && (Y <= 1), bool >::type = true>
FORCEINLINE float64x2_t VectorSwizzleImpl2(vector4d Vec)
{
	return __builtin_shufflevector(Vec.xy, Vec.xy, X, Y);
}

template <int X, int Y, typename std::enable_if < (X <= 1) && (Y > 1), bool >::type = true>
FORCEINLINE float64x2_t VectorSwizzleImpl2(vector4d Vec)
{
	return __builtin_shufflevector(Vec.xy, Vec.zw, X, Y);
}

template <int X, int Y, typename std::enable_if < (X > 1) && (Y <= 1), bool >::type = true>
FORCEINLINE float64x2_t VectorSwizzleImpl2(vector4d Vec)
{
	return __builtin_shufflevector(Vec.zw, Vec.xy, X - 2, Y + 2);
}

template <int X, int Y, typename std::enable_if < (X > 1) && (Y > 1), bool >::type = true>
FORCEINLINE float64x2_t VectorSwizzleImpl2(vector4d Vec)
{
	return __builtin_shufflevector(Vec.zw, Vec.zw, X - 2, Y);
}

template <int X, int Y, int Z, int W>
FORCEINLINE vector4d VectorSwizzleImpl(vector4d Vec)
{
	vector4d r;
	r.xy = VectorSwizzleImpl2<X, Y>(Vec);
	r.zw = VectorSwizzleImpl2<Z, W>(Vec);
	return r;
}



template <int X, int Y, int Z, int W>
FORCEINLINE vector4f VectorShuffleImpl(vector4f Vec1, vector4f Vec2)
{
	return __builtin_shufflevector(Vec1, Vec2, X, Y, Z + 4, W + 4);
}

template <int X, int Y, int Z, int W>
FORCEINLINE vector4d VectorShuffleImpl(vector4d Vec1, vector4d Vec2)
{
	vector4d r;
	r.xy = VectorSwizzleImpl2<X, Y>(Vec1);
	r.zw = VectorSwizzleImpl2<Z, W>(Vec2);
	return r;
}

template <int ElementIndex>
FORCEINLINE vector4f VectorReplicateImpl(const vector4f& Vec)
{
	return vdupq_n_f32(vgetq_lane_f32(Vec, ElementIndex));
}

template <int ElementIndex>
FORCEINLINE float64x2_t VectorReplicateImpl(const float64x2_t& Vec)
{
	return vdupq_n_f64(vgetq_lane_f64(Vec, ElementIndex));
}

template <int ElementIndex, typename std::enable_if < (ElementIndex <= 1), bool >::type = true >
FORCEINLINE vector4d VectorReplicateImpl(const vector4d& Vec)
{
	vector4d r;
	r.xy = VectorReplicateImpl<ElementIndex>(Vec.xy);
	r.zw = r.xy;
	return r;
}

template <int ElementIndex, typename std::enable_if < (ElementIndex > 1), bool >::type = true >
FORCEINLINE vector4d VectorReplicateImpl(const vector4d& Vec)
{
	vector4d r;
	r.zw = VectorReplicateImpl<ElementIndex - 2>(Vec.zw);
	r.xy = r.zw;
	return r;
}

#define VectorReplicate( Vec, ElementIndex ) VectorReplicateImpl<ElementIndex>(Vec)
#define VectorSwizzle( Vec, X, Y, Z, W ) VectorSwizzleImpl<X, Y, Z, W>(Vec)
//#define VectorSwizzle2(Vec, X, Y) VectorSwizzleImpl2<X, Y>(Vec)
#define VectorShuffle( Vec1, Vec2, X, Y, Z, W )	VectorShuffleImpl<X, Y, Z, W>(Vec1, Vec2)



#else
#pragma error("vector swizzle not implement here!");

#endif



RTM_IMPL_VERSION_NAMESPACE_END
}

RTM_IMPL_FILE_PRAGMA_POP

