function conjugatepartition(p::AbstractVector{T}) where T <: Integer
    isempty(p) && throw(ArgumentError("Empty partition"))
    any(iszero.(p)) && throw(ArgumentError("Partition contains zeros"))
    (length(p) == 1) && return ones(T, first(p))
    (all(isone.(p))) && return [T(length(p))]

    p = sort(p, rev=true)
    result = T[]
    j = length(p)
    while true
        result = T[result; j]
        while length(result) >= p[j]
            j -= 1
            if j == 0
                return result
            end
        end
    end
end

function randompartitionAD5(n, m=nothing; rng=Random.GLOBAL_RNG)
    """
    Uniform sampling of partitions of n using PDC with deterministic second half
    Algorithm 5 in Arratia & DeSalvo 2011 arXiv:1110.3856v7 and DeSalvo's answer at
    http://stackoverflow.com/questions/2161406/how-do-i-generate-a-uniform-random-integer-partition
    
    Question: How do I generate a uniform random integer partition?
    Stephen DeSalvo answered Nov 7 '13 at 6:46

    The title of this post is a bit misleading. A random integer partition is by
    default unrestricted, meaning it can have as many parts of any size. The
    specific question asked is about partitions of n into m parts, which is a
    type of restricted integer partition.

    For generating unrestricted integer partitions, a very fast and simple
    algorithm is due to Fristedt, in a paper called The Structure of Random
    Partitions of Large Integer (1993). The algorithm is as follows:

    Set x = exp(-pi/sqrt(6n) ). Generate independent random variables Z(1),
    Z(2), ..., Z(n), where Z(i) is geometrically distributed with parameter
    1-x^i. IF sum i*Z(i) = n, where the sum is taken over all i=1,2,...,n, then
    STOP. ELSE, repeat 2. Once the algorithm stops, then Z(1) is the number of
    1s, Z(2) is the number of 2s, etc., in a partition chosen uniformly at
    random. The probability of accepting a randomly chosen set of Z's is
    asymptotically 1/(94n^3)^(1/4), which means one would expect to run this
    algorithm O(n^(3/4)) times before accepting a single sample.

    The reason I took the time to explain this algorithm is because it applies
    directly to the problem of generating a partition of n into exactly m parts.
    First, observe that

    The number of partitions of n into exactly m parts is equal to the number of
    partitions of n with largest part equal to m.

    Then we may apply Fristedt's algorithm directly, but instead of generating
    Z(1), Z(2), ..., Z(n), we can generate Z(1), Z(2), ..., Z(m-1), Z(m)+1 (the
    +1 here ensures that the largest part is exactly m, and 1+Z(m) is equal in
    distribution to Z(m) conditional on Z(m)>=1) and set all other Z(m+1),
    Z(m+2), ... equal to 0. Then once we obtain the target sum in step 3 we are
    also guaranteed to have an unbiased sample. To obtain a partition of n into
    exactly m parts simply take the conjugate of the partition generated.

    The advantage this has over the recursive method of Nijenhuis and Wilf is
    that there is no memory requirements other than to store the random
    variables Z(1), Z(2), etc. Also, the value of x can be anything between 0
    and 1 and this algorithm is still unbiased! Choosing a good value of x,
    however, can make the algorithm much faster, though the choice in Step 1 is
    nearly optimal for unrestricted integer partitions.

    If n is really huge and Fristedt's algorithm takes too long (and table
    methods are out of the question), then there are other options, but they are
    a little more complicated; see my thesis
    https://sites.google.com/site/stephendesalvo/home/papers for more info on
    probabilistic divide-and-conquer and its applications.
    """

    if n < 1
        throw(ArgumentError("n should be greater or equal to 1"))
    end
    if m !== nothing && !(1 <= m <= n)
        throw(ArgumentError("Number of parts m should satisfy 1 <= m <= n"))
    end
    if n == 1
        return [1]
    end
    if m == 1
        return [n]
    end

    idx_range = 2:n
    while true
        ps = exp.(log1mexp.(-idx_range .* π / sqrt(6 * n)))
        # check_args=false so that it does not fail on some ps[k] = 1.0
        # it will fail if some ps[k] = 0.0 but this is never the case
        # because the argument of log1mexp is never 0.
        Z = rand.(rng, Geometric.(ps, check_args=false))

        if m !== nothing
            Z[m:end] .= 0
            Z[m-1] += 1
        end

        k = n - sum(idx_range .* Z)
        if k >= 0 && rand(rng) < exp(-k * π / sqrt(6 * n))
            Z = [k; Z]
            partition = reverse([i for (i, zi) in zip(1:n, Z) for _ in 1:zi])
            return m === nothing ? partition : conjugatepartition(partition)
        end
    end
end
