module DimArrays

const NameEl = Union{Symbol,Void}
const FuncEl = Union{Function,Dict,Void}

struct DimArray{T,N,TT} <: AbstractArray{T,N}
    array::TT
    dnames::NTuple{N,NameEl}    ## dimension names: symbol or nothing
    ifuncs::NTuple{N,FuncEl}    ## index functions: dictionary if bounded
    cname::NameEl               ## content name, if any
end

const DimVector{T} = DimArray{T,1}
const DimMatrix{T} = DimArray{T,2}
const DimVecOrMat{T} = Union{DimVector{T},DimMatrix{T}}
const DimRowVector{T} = DimArray{T,2,<:RowVector{T}}


### Getters ###
###############

naked(a::DimArray) = a.array
naked(a::AbstractArray) = a

dname(a::DimArray, d::Int) = a.dnames[d]==nothing ? defname(d) : a.dnames[d]
dname(a::AbstractArray, d::Int) = defname(d)
dnames(a::AbstractArray) = [ dname(a,d) for d=1:ndims(a) ]

function defname(d::Int)
    d==1 && return :row
    d==2 && return :col
    d==3 && return :page
    d==4 && return Symbol(:dim,d)
end

ifunc(a::DimArray, d::Int) = a.ifuncs[d]==nothing ? identity : a.ifuncs[d]
ifunc(a::AbstractArray, d::Int) = identity
ifuncs(a::AbstractArray) = [ ifunc(a,d) for d=1:ndims(a) ]

label(a::DimArray) = a.cname==nothing ? :content : a.cname
haslabel(a::DimArray) = a.cname != nothing

using Lazy: @forward
@forward DimArray.array  Base.size, Base.length, Base.ndims, Base.getindex, Base.setindex!, Base.append!
@forward DimArray.array  Base.first, Base.start, Base.next, Base.done


### Constructors ###
####################

export DimArray, DimVector, DimMatrix

ensurefunc(f::Function) = f
ensurefunc(f::Dict) = i -> f[i]
ensurefunc(f::Number) = f==1 ? identity : (i -> i*f)

using Base.Iterators: repeated

const FuncOrNumberOrDict = Union{Function, Number, Dict}
const SymbolOrString = Union{Symbol, String}

"""
    DimArray(x, names...)
    DimArray(x, names..., maps...)
Interprets symbols & strings as dimension names (in order), and numbers, functions & dictionaries as index maps (for each dimension).
Numbers are converted to functions `i -> i*n` to indicate e.g. that we have every n-th data point.
The `ndims(x)+1`-th name (if given) is equivalent to keyword `label=name`, labelling what the elements of `x` mean.
Giving too few names/maps is no problem, too many will give a warning.
"""
function DimArray(x::TT, tup::Vararg; label = nothing) where TT<:AbstractArray{T,N} where {T,N}
    names = Vector{NameEl}(repeated(nothing, N) |> collect)
    funcs = Vector{FuncEl}(repeated(nothing, N) |> collect)
    n=0; f=0;
    for z in tup
        if isa(z, FuncOrNumberOrDict)
            if f<N
                funcs[f += 1] = ensurefunc(z)
            else
                warn("DimArray is ignoring function $z")
            end
        elseif isa(z, SymbolOrString)
            if n<N
                if z!=""
                    names[n += 1] = Symbol(z)
                else
                    n += 1
                end
            elseif n==N && z!=""
                label=Symbol(z)
                n += 1
            elseif z!=""
                warn("DimArray is ignoring name $z")
            end
        elseif isa(z, Vector)
            warn("DimArray is ignoring vector $z, this shouldn't happen!")
        elseif isa(z, Tuple)
            warn("DimArray is ignoring tuple $z, this really shouldn't happen!")
        elseif z==nothing
            ## ignore, no problem!
        end
    end
    DimArray{T,N,TT}(x, NTuple{N,NameEl}(names), NTuple{N,FuncEl}(funcs), label)
end

"""
    DimArray(x, [n1, n2, ...], [f1, f2, ...])
    DimArray(x, (n1, n2, ...), (f1, f2, ...), label)
Vectors of names/maps are splatted and digested as above.
"""
DimArray(x::AbstractArray, list::Vector; kw...) = DimArray(x, list...; kw...);
DimArray(x::AbstractArray, list::Vector, label::Symbol; kw...) = DimArray(x, list...; label=label, kw...);

DimArray(x::AbstractArray, a::Vector, b::Vector; kw...) = DimArray(x, a..., b...; kw...)
DimArray(x::AbstractArray, a::Vector, b::Vector, label::Symbol; kw...) = DimArray(x, a..., b...; label=label, kw...)

DimVector(x::AbstractVector, rest...; kw...) = DimArray(x, rest...; kw...)
DimMatrix(x::AbstractVector, rest...; kw...) = DimArray(reshape(x,:,1), rest...; kw...)
DimMatrix(x::AbstractMatrix, rest...; kw...) = DimArray(x, rest...; kw...)

DimArray(s::Symbol=:row) = DimArray([], s)
DimVector(s::Symbol=:row) = DimArray([], s)

## avoid the above splatting for perfectly formed case, for functions below -- will fail on wrong-length tuples
DimArray(x::TT, dnames::Tuple, ifuncs::Tuple, cname::NameEl) where TT <: AbstractArray{T,N} where {T,N} =
    DimArray{T,N,TT}(x, dnames, ifuncs, cname);


### One-array operations ###
############################

import Base: push!, map, map!

push!(a::DimVector, x) = begin push!(a.array, x); a end

map!(f::Function, a::DimVector) = begin map!(f, a.array); a end
map(f::Function, a::DimVector) = DimArray(map(f, a.array), a.dnames, a.ifuncs, a.cname)

import Base: transpose, ctranspose, permutedims

for op in (:transpose, :ctranspose)
    @eval begin
        ($op)(a::DimMatrix) = DimArray( ($op)(a.array), rev2(a.dnames), rev2(a.ifuncs), a.cname)
        ($op)(a::DimVector) = DimArray( ($op)(a.array), (:transpose, a.dnames[1]), (nothing, a.ifuncs[1]), a.cname)
        ($op)(a::DimRowVector) = DimArray( ($op)(a.array), (a.dnames[2],), (a.ifuncs[2],), a.cname)
    end # @eval
end

rev2(tup::Tuple) = (tup[2], tup[1])

function permutedims(a::DimArray, p::Vector{Int})
    @assert ndims(a)==length(p) "no valid permutation of dimensions"
    DimArray(permutedims(a.array, p), a.dnames[p], a.ifuncs[p], a.cname)
end

permutedims(a::DimMatrix) = permutedims(a, [2,1])

import Base: size, squeeze, slicedim

function size(a::DimArray, d::Union{Int,Symbol})
    d = ensuredim(a,d)
    size(a.array, d)
end

function squeeze(a::DimArray, d::Union{Int,Symbol}; verbose=false, zerodim=false)
    d = ensuredim(a,d)
    d==1 && ndims(a)==1 && !zerodim && return a[1]
    out = DimArray(Base.squeeze(a.array, d), dropd(a.dnames,d), dropd(a.ifuncs,d), a.cname)
    verbose && info("""squeezed along :$(dname(a,d)), leaving directions $(dnames(out)) size $(size(out))""")
    return out
end

function squeeze(a::DimArray, dims::AbstractVector; kw...)
    length(dims)==1 && return squeeze(a, dims[1]; kw...)
    # dims = sort(dims)
    return squeeze(squeeze(a, dims[end]; kw...), dims[1:end-1]; kw...)
end

function squeeze(a::DimArray; kw...)
    which = [d for d=1:ndims(a) if size(a,d)==1]
    squeeze(a, which; kw...)
end

function slicedim(a::DimArray, d::Union{Symbol, Int}, i::Int)
    d = ensuredim(a,d)
    DimArray(slicedim(a.array,d,i), dropd(a.dnames,d), dropd(a.ifuncs,d), a.cname)
end

ensuredim(a::DimArray, d::Int) = d
function ensuredim(a::DimArray{T,N}, s::Symbol) where {T,N}
    d = findfirst(dnames(a), s)
    1<=d<=N || error("""can't use direction :$s; valid options are $(dnames(a))""")
    return d
end

dropd(vec::Vector, d) = d>length(vec) ? vec : append!(vec[1:d-1], vec[d+1:end])
dropd(tup::Tuple, d) = d>length(tup) ? tup : (tup[1:d-1]..., tup[d+1:end]...)

import Base: sum, mean, std, maximum, minimum

for op in (:sum, :mean, :std, :maximum, :minimum )
    @eval begin

        function ($op)(a::DimArray, d::Int; squeeze=false, verbose=false, kw...)

            data = ($op)(a.array, d; kw...)
            if squeeze
                out = DimArray(Base.squeeze(data, d), dropd(a.dnames,d), dropd(a.ifuncs,d), a.cname)
            else
                out = DimArray(data, a.dnames, a.ifuncs, a.cname)
            end

            verbose && info("""$(($op))-ed along :$(dname(a,d)), leaving directions $(dnames(out)) size $(size(out))""")
            return out
        end

        function ($op)(a::DimArray, s::Symbol; squeeze=true, verbose=false, kw...) ## different defaults!
            d = ensuredim(a,s)
            return ($op)(a, d; squeeze=squeeze, verbose=verbose, kw...)
        end

        function ($op)(a::DimArray, dims::AbstractVector; kw...)
            length(dims)==1 && return ($op)(a, dims[1]; kw...)
            # dims = sort(dims)
            b = ($op)(a, dims[end]; kw...)
            return ($op)(b, dims[1:end-1]; kw...)
        end

    end # @eval
end


### Scalar operations ###
#########################

import Base: +, -, *, /, \

-(a::DimArray) = DimArray(-a.array, a.dnames, a.ifuncs, a.cname)

for op in (:+, :-, :*)
    @eval begin
        ($op){T1<:Number,T2<:Number}(a::DimArray{T1}, s::T2) = DimArray(($op)(a.array, s), a.dnames, a.ifuncs, a.cname)
        ($op){T1<:Number,T2<:Number}(s::T1, a::DimArray{T2}) = DimArray(($op)(s, a.array), a.dnames, a.ifuncs, a.cname)
    end
end
/{T1<:Number,T2<:Number}(a::DimArray{T1}, s::T2) = DimArray(a.array / s, a.dnames, a.ifuncs, a.cname)
\{T1<:Number,T2<:Number}(s::T1, a::DimArray{T2}) = DimArray(s \ a.array, a.dnames, a.ifuncs, a.cname)


### Generators ###
##################

Base.collect(itr::Base.Generator{<:DimArray}) =
    DimArray(collect(Base.Generator(itr.f, itr.iter.array)) , itr.iter.dnames, itr.iter.ifuncs, itr.iter.cname)


### Broadcasting ###
####################

Base.Broadcast._containertype(::Type{<:DimArray}) = DimArray

Base.Broadcast.promote_containertype(::Type{DimArray}, _) = DimArray
Base.Broadcast.promote_containertype(_, ::Type{DimArray}) = DimArray
Base.Broadcast.promote_containertype(::Type{DimArray}, ::Type{Array}) = DimArray
Base.Broadcast.promote_containertype(::Type{Array}, ::Type{DimArray}) = DimArray
Base.Broadcast.promote_containertype(::Type{DimArray}, ::Type{DimArray}) = DimArray

array(a::DimArray) = a.array
array(x) = x

function Base.Broadcast.broadcast_c(fun, t::Type{DimArray}, list...)
    res = broadcast(fun, array.(list)...)
    T = eltype(res); N = ndims(res); TT = typeof(res)
    ## is there a DimArray of the final shape, and not made by v'?
    for a in list
        if isa(a, DimArray) && size(a) == size(res) && findfirst(dnames(a), :transpose)==0
        return DimArray{T,N,TT}(res, a.dnames, a.ifuncs, a.cname) end
    end
    ## if not, you have to work harder:
    rest = reconcile(list, N)
    DimArray{T,N,TT}(res, rest...)
end

function reconcile(list, N)
    names = Vector{NameEl}(repeated(nothing, N) |> collect)
    funcs = Vector{FuncEl}(repeated(nothing, N) |> collect)
    label = nothing
    n=0; f=0;
    for a in reverse(list) ## overwrite with earliest names
        if isa(a, DimArray)
            for d=1:ndims(a)
                if ifunc(a,d) !== identity ## then adopt this one
                    funcs[d] = a.ifuncs[d]
                end
                if a.dnames[d] !== nothing && a.dnames[d] !== :transpose
                    names[d] = a.dnames[d]
                end
            end
            a.cname !== nothing && (label = a.cname)
        end
    end
    for d=2:N
        if names[d] != nothing && findfirst(names[1:d-1], names[d])>0 ## if symbol already used, add a prime...
            for dd=N:-1:d ## ... to all later occurances too
                if names[dd]==names[d]
                    names[dd] = Symbol(names[d], "′")
                end
            end
        end
    end
    NTuple{N,NameEl}(names), NTuple{N,FuncEl}(funcs), label
end


### Two-array operations ###
############################

import Base: append!, hcat, vcat, cat

for op in (:+, :-, :append!)
    @eval begin
        ## adding a boring array
        ($op){T1<:Number,T2<:Number,N}(a::DimArray{T1,N}, b::AbstractArray{T2,N}) =
            DimArray(($op)(a.array, b), a.dnames, a.ifuncs, a.cname)

        ($op){T1<:Number,T2<:Number,N}(b::AbstractArray{T1,N}, a::DimArray{T2,N}) =
            DimArray(($op)(b, a.array), a.dnames, a.ifuncs, a.cname)

        ## adding two DimArrays
        ($op){T1<:Number, T2<:Number}(a::DimArray{T1}, b::DimArray{T2}) =
            DimArray(($op)(a.array, b.array), reconcile((a,b),ndims(a))...)
    end # @eval
end

function hcat(a::DimArray, bb::Vararg{DimArray})
    out = hcat(array(a), array.(bb)...)
    DimArray(out, reconcile((a,bb...), ndims(out))...)
end

function vcat(a::DimArray, bb::Vararg{DimArray})
    out = vcat(array(a), array.(bb)...)
    DimArray(out, reconcile((a,bb...), ndims(out))...)
end

import Base: *, kron

function *(a::DimArray, b::DimArray)
    a.dnames[end]!=b.dnames[1] && a.dnames[end]!=nothing && b.dnames[1]!=nothing &&
        warn("multiplying along dimensions with mismatched names", once=true)
    names = (a.dnames[1:end-1]..., b.dnames[2:end]...)
    funcs = (a.ifuncs[1:end-1]..., b.ifuncs[2:end]...)
    DimArray(a.array * b.array, names, funcs, a.cname )
end

*(a::DimArray, b::AbstractMatrix) = *(a, DimArray(b))
*(a::DimArray, b::AbstractVector) = *(a, DimArray(b))
*(a::AbstractMatrix, b::DimArray) = *(DimArray(a), b)

function kron(a::DimMatrix, b::DimMatrix)
    names = prodsomething.(a.dnames, b.dnames)
    funcs = firstsomething.(a.ifuncs, b.ifuncs)
    DimArray(kron(a.array, b.array), names, funcs, a.cname)
end

function prodsomething(list...)
    out = ""
    for s in list
        s != nothing && (out *= "_"*string(s))
    end
    out == "" && return nothing
    return Symbol(out[2:end])
end

function firstsomething(list...)
    for f in list
        f != nothing && return f
    end
    return nothing
end


### Nested Arrays ###
#####################

export nest

"""
    nest([A1, A2, ...])
    nest([A1, A2, ...], sym)
Creates an array with `ndims(A1)+1` dimensions. All the constituent arrays should be similar!
Optional 2nd argument names the new dimension, giving `DimArray(nest(...),sym)`;
this also happens if `A1 isa DimArray`.

    [A1, A2, ...] |> nest(s)
Gives a function which acts on vectors as above.
"""
nest(sym::Symbol) = vec -> nest(vec, sym)

nest(vec::AbstractVector, sym::Symbol) = nest(DimArray(vec,sym))

function nest(vec::AbstractVector) ## Allow Vector{Any} but assume contents is all the same Array
    x1 = first(vec)
    isa(x1,AbstractArray) || return vec ## nothing to do

    out = similar(x1, size(x1)..., length(vec), )
    for (i, x) in enumerate(vec)
        out[batchindex(out,i)...] = x
    end

    if isa(vec, DimArray) && isa(x1, DimArray)
        return DimArray(out, (x1.dnames..., vec.dnames[1]), (x1.ifuncs..., vec.ifuncs[1]), x1.cname)

    elseif isa(vec, DimArray) ## and "x1" is not -- use nothings
        none = repeated(nothing, ndims(x1)) |> collect
        return DimArray(out, (none..., vec.dnames[1]), (none..., vec.ifuncs[1]), nothing)

    elseif isa(x1, DimArray) ## and "vec" is not -- use "nest"
        return DimArray(out, (x1.dnames..., :nest), (x1.ifuncs..., nothing), x1.cname)

    else ## no DimArrays at all
        return out
    end
end

## that's based on Flux.batch
batchindex(xs, i) = (reverse(Base.tail(reverse(indices(xs))))..., i)

function nest(arr::AbstractArray{D}) where {D}
    one = [nest(slicedim(arr, ndims(arr), c)) for c=1:size(arr,ndims(arr))]
    nest(one)
end


### Display ###
###############

## TODO change to extending summary()

function Base.showarray(io::IO, a::DimArray, repr::Bool = true; header = true)
    if repr ## compact
        print(io, "DimArray(")
        Base.showarray(io, a.array, true)
        for d=1:ndims(a)
            print(io, ", :",dname(a,d))
        end
        haslabel(a) && print(io, ", :", label(a))
        print(io, ")")
    else ## full
        lab = haslabel(a) ? ", label = "*string(label(a))*"," : ""
        if typeof(a.array) <: Array
            ndims(a)==0 && println(io, "DimArray{",eltype(a),"}$lab of zero dimensions:")
            ndims(a)==1 && println(io, "DimVector{",eltype(a),"}$lab with dimension:")
            ndims(a)==2 && println(io, "DimMatrix{",eltype(a),"}$lab with dimensions:")
            ndims(a)>=3 && println(io, "DimArray{",eltype(a),",",ndims(a),"}$lab with dimensions:")
        elseif typeof(a.array) <: RowVector
            println(io, "DimRowVector{",eltype(a),"} with dimensions:")
        else
            println(io, summary(a.array), " wrapped in a DimArray with:")
        end
        for d=1:ndims(a)
            if a.dnames[d]==nothing #|| a.dnames[d]==:transpose
                        print("   ⭒ ") ## different symbol for default names
            elseif d==1 print(io, "   ⬙ ") ## ⇁⇂
            elseif d==2 print(io, "   ⬗ ")
            else        print(io, "   ◇ ")
            end
            print(io, rpad(string(dname(a,d)),4)," = ", stringfew(size(a,d)) )
            if ifunc(a,d)!=identity
                println(io, "  ⟹   ", stringfew(size(a,d), ifunc(a,d)))
            else
                println(io, "")
            end
        end
        if eltype(a) <: Union{Number, String, Char}
            ndims(a)>=3 && println("")
            Base.showarray(io, a.array, false; header=false)
        elseif length(a) >= 1
            println(io, "The first element is [", repeated("1,", ndims(a)-1)...,"1] = " )
            show(io,"text/plain", first(a))
        end
    end
end

function stringfew(n, f=identity)
    n<=5 && return join(stringone.(collect(1:n),f), ", ")
    return join(stringone.([1,2],f), ", ") * ", … " * join(stringone.([n-1,n],f), ", ")
end

function stringone(n, f=identity)
    f(n) isa Integer && return string(f(n))
    f(n) isa Number && return string(round(f(n),3))
    return string(f(n))
end


### Plotting ###
################

using RecipesBase

@recipe function ff(a::DimVector)
    label --> ""

    ## first dim is x axis
    if a.dnames[1] != nothing
        xaxis --> string(dname(a,1))
    end
    xlist = ifunc(a,1).(1:length(a))

    xlist, a.array
end

@recipe function ff(a::DimMatrix)

    ## first dim is x axis
    xaxis --> string(dname(a,1))
    xlist = ifunc(a,1).(1:size(a,1))

    ## second dim labels series
    ylist = string(dname(a,2))*" = " .* string.(ifunc(a,2).(1:size(a,2)))
    label --> reshape(ylist, 1,:) ## because series labels must be cols of a matrix, not el of vector

    if haslabel(a)
        yaxis --> string(a.cname)
    end

    xlist, a.array
end


### Conversions ###
###################

using Requires

@require NamedArrays begin
    using NamedArrays: NamedArray

    axestuple(a::DimArray) = Tuple([f.(collect(1:size(a,d))) for (d,f) in enumerate(ifuncs(a))])
    nametuple(a::DimArray) = Tuple(dnames(a))
    Base.convert(::Type{NamedArray}, a::DimArray) = NamedArray(a.array, axestuple(a), nametuple(a))
    NamedArray(a::DimArray) = convert(NamedArray, a)

    ## TODO reverse conversion?
end

@require AxisArrays begin
    using AxisArrays: AxisArray, Axis

    axislist(a::DimArray) = [ Axis{dname(a,d)}( ifunc(a,d).(collect(1:size(a,d)) ) ) for d=1:ndims(a) ]
    Base.convert(::Type{AxisArray}, a::DimArray) = AxisArray(a.array, axislist(a)...)
    AxisArray(a::DimArray) = convert(AxisArray, a)

    ## TODO reverse conversion?
end


### End ###
###########

end # module
