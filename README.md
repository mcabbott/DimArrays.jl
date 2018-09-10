# DimArrays

I'm trying to upgrade this to Julia 0.7. But it has some errors still.

## v1 Explanation

Sometimes I collect data as a vector of matrices, or a vector of those, and sometimes I smash that into a 4-tensor.
Then I make lots of mistakes about which dimensions are which.
So because I had more important things to do, I thought I'd write a simple package to remember their names for me.

```julia
using DimArrays

list = [];
for i=1:33
    slowcalc = sqrt(i) .* randn(3,13) .+ i
    push!(list, DimArray(slowcalc, :a, :b, :c ))  ## add labels for 1st and 2nd dimensions  
end

list3 = nest(list, :iter)  ## now i is the 3rd index, and named "iter"

mean(list3, :iter)  ## equivalent to squeeze(mean(list3,3),3)
```

And for quick-and-dirty plots, I want the axes & series labelled
so that I know which is which, and can quickly decide if I need a transpose:
```julia
using Plots

plot(slicedim(list3, :b, 1)' , legend=:bottomright)
```
Here `slicedim(list3, :b, 1) == list3[:,1,:]` in contents, but retains the labels.

Besides each dimension's name (a Symbol, strings will be converted) it can also store a function,
which is used in plotting to scale the axes etc. (But only the output, `getindex` uses original integer indices).
You can pass a number by which to scale the index, or a dictionary, instead of a function.
For example, this plots data sampled every 4 iterations correctly over the above:
```julia
saveevery = 4
list4 = DimArray([], :iter, saveevery);  ## equivalent to function  i->4i
for i=1:33
    slowcalc = sqrt(i) .* randn(3,23) .+ i
    slownice = DimArray(slowcalc, [:a, :b], [Dict(1=>"one", 2=>"two", 3=>"three")], :stuff )
                                            ## equivalent to  i->Dict(...)[i]
    rem(i,saveevery)==0 && push!(list4, slownice)
end
nest(list4)

plot!(mean(nest(list4), :b)', s=:dash)
```
If you do not provide a name for a dimension (or give an empty string "") then you can still refer to it
by default names like `size(x, :row) == size(x,1)` or `maximum(y, :col)` etc. However these defaults are not stored,
and not manipulated by `transpose(x)` or `kron(x,y)`.

For now, the list of functions supported is:

* `sum, mean, std, maximum, minimum, squeeze`: all can be called with a dimension's name.
    and then squeeze that dimension, like `mean(..., :b)` above.
    They can also be called with a list of dimensions: `sum(x, [1,:c])` etc.
* `slicedim, size` understand a dimension's name.
* `push!, append!, hcat, vcat, transpose, ctranspose, permutedims`.
* `broadcast`, both by scalars `DimArray(rand(3,3)) .+ 10`
    and for combining arrays `DimArray(rand(3),"rand col") .* DimArray(ones(1,4),"", "one row")`.
* Matrix multiplication `*` will warn (once) if you multiply along directions with mismatched names...
    unsure if that's a good idea?
    And `kron`ecker products produce new names like `:a_b`.  
* `collect`, implicitly used by comprehensions like `[ sqrt(n) for n in DimVector(1:10, "int")' ]`.

Since `DimArray <: AbstractArray` anything else will fall back on their methods,
and forget the dimension labels. Functions exported by `using DimArrays` are:

* `DimArray`, `DimVector`, `DimMatrix`, `nest`.

Of course I'm not the first person to have this idea, nor to write up his own package...
as I discovered half way into writing this. Both [NamedArrays](https://github.com/davidavdav/NamedArrays.jl)
and [AxisArrays](https://github.com/JuliaArrays/AxisArrays.jl) do something similar.
This package is lighter-weight, and implements a few things I wanted like `push!` (both of the others are immutable),
and `plot` recipes, but makes little attempt to be high-performance.
You can `convert` to these via `AxisArray(a::DimArray)` and `NamedArray(a::DimArray)`, preserving axis (dimension) names.

For other views on vectors of arrays etc, see  [RecursiveArrayTools](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl)
and [JuliennedArrays](https://github.com/bramtayl/JuliennedArrays.jl).

ToDo:
* Make things like `x[:, 1:10:end]` update the functions.
* Figure out Julia 0.7's new broadcasting machinery.

Michael Abbott,
January 2018, mostly.


[![Build Status](https://travis-ci.org/mcabbott/DimArrays.jl.svg?branch=master)](https://travis-ci.org/mcabbott/DimArrays.jl)

[![Coverage Status](https://coveralls.io/repos/mcabbott/DimArrays.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mcabbott/DimArrays.jl?branch=master)

[![codecov.io](http://codecov.io/github/mcabbott/DimArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/mcabbott/DimArrays.jl?branch=master)

<!--
Note to self:
pandoc -o README.html README.md
-->
