using DimArrays
using Base.Test

a = DimVector([1,2,3,4], :aa, 100)

@test DimArrays.dname(a,1) == :aa
@test DimArrays.ifunc(a,1)(1) == 100
@test DimArrays.haslabel(a) == false

@test typeof(a') <: DimMatrix
@test ndims( (a').' )==1
@test DimArrays.dname( (a').' ,1) == :aa

@test typeof( map(sqrt, a) ) <: DimVector
@test typeof( [sqrt(i) for i in a] ) <: DimVector

b = DimArray([1.0 2; 3 4])

@test string(b) == "DimArray([1.0 2.0; 3.0 4.0], :row, :col)"

@test sum(b, [:row,:col])[1] ≈ 10
@test std(b, :col)[1] ≈ 0.7071067811865476
@test typeof( mean(b,2) ) <: DimMatrix

@test slicedim(b, :col, 1)[2] ≈ 3
@test typeof( slicedim(b, 2, 1) ) <: DimVector

@test (b .+ 99)[1,1] ≈ 100
@test typeof(b .+ 99) <: DimMatrix

@test sum( hcat(DimArray([1,3]), [2,4]) ./ b ) ≈ 4

@test maximum( nest([DimArray([1,3]), [2,4]]) .- b ) ≈ 0
