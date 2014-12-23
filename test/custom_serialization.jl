module MyTypes

export MyType, MyContainer

## Objects we want to save
# data in MyType is always of length 5, and that is the basis for a more efficient serialization
immutable MyType{T}
    data::Vector{T}
    id::Int
    
    function MyType(v::Vector{T}, id::Integer)
        length(v) == 5 || error("All vectors must be of length 5")
        new(v, id)
    end
end
MyType{T}(v::Vector{T}, id::Integer) = MyType{T}(v, id)
Base.eltype{T}(::Type{MyType{T}}) = T
==(a::MyType, b::MyType) = a.data == b.data && a.id == b.id

immutable MyContainer{T}
    objs::Vector{MyType{T}}
end
Base.eltype{T}(::Type{MyContainer{T}}) = T
==(a::MyContainer, b::MyContainer) = length(a.objs) == length(b.objs) && all(i->a.objs[i]==b.objs[i], 1:length(a.objs))

end


### Here are the definitions needed to implement the custom serialization
module MySerializer

using HDF5, JLD, MyTypes
# The following imports are simply to use them, not to extend them
import JLD: JldFile, JldWriteSession, JldTypeInfo

## Defining the serialization format
type MyContainerSerializer{T}
    data::Matrix{T}
    ids::Vector{Int}
end
Base.eltype{T}(::Type{MyContainerSerializer{T}}) = T

# A necessary stub function
JLD.h5fieldtype{T<:MyContainer}(parent::JldFile, ::Type{T}, commit::Bool) =
    JLD.h5type(parent, T, commit)

# Specify that a MyContainer will be written in the format specified by MyContainerSerializer
function JLD.h5type{T<:MyContainer}(parent::JldFile, ::Type{T}, commit::Bool)
    haskey(parent.jlh5type, T) && return parent.jlh5type[T]
    Tser = MyContainerSerializer{eltype(T)}
    JLD.h5type(parent, Tser, commit)
end

# The next two functions overwrite the standard auto-generated methods for these types
function JLD.gen_h5convert{T<:MyContainer}(parent::JldFile, ::Type{T})
    # Create the default "write converter" for the MyContainerSerializer type corresponding to T
    Tser = MyContainerSerializer{eltype(T)}
    JLD.gen_h5convert(parent, Tser)
    # Create the default "read converter" for a MyContainerSerializer,
    # but give it a different name so we can call it independently
    JLD._gen_jlconvert_type(JldTypeInfo(parent, Tser, true), Tser, :_jlconvert)
end
# Prevent the auto-generation of the typically-named "read converter" for a MyContainerSerializer
# This will allow our custom jlconvert method below to be called instead
JLD.gen_jlconvert{T<:MyContainerSerializer}(::JldTypeInfo, ::Type{T}) = nothing

# This function implements the conversion to serialized format.
# First we convert to a MyContainerSerializer object, then we write this object in
# the standard format for such objects
function JLD.h5convert!{T<:MyContainer}(out::Ptr, file::JldFile, container::T, writesession::JldWriteSession)
    ids = [obj.id for obj in container.objs]
    n = length(container.objs)
    vectors = Array(eltype(T), 5, n)
    for i = 1:n
        vectors[:,i] = container.objs[i].data
    end
    serdata = MyContainerSerializer{eltype(T)}(vectors, ids)
    JLD.h5convert!(out, file, serdata, writesession)
end

# The "read converter", which unpacks the serialized format into the one we want to return
function JLD.jlconvert{T<:MyContainerSerializer}(::Type{T}, file::JldFile, ptr::Ptr)
    # Read the serialized object
    serdata = JLD._jlconvert(MyContainerSerializer{eltype(T)}, file, ptr)
    # Convert to a MyContainer
    n = length(serdata.ids)
    MyContainer{eltype(T)}([MyType(serdata.data[:,i], serdata.ids[i]) for i = 1:n])
end

end   # MySerializer



using MyTypes, JLD, Base.Test

obj1 = MyType(rand(5), 2)
obj2 = MyType(rand(5), 17)
container = MyContainer([obj1,obj2])
filename = joinpath(tempdir(), "customserializer.jld")
jldopen(filename, "w") do file
    write(file, "mydata", container)
end

container_r = jldopen(filename) do file
    read(file, "mydata")
end

@test container_r == container
