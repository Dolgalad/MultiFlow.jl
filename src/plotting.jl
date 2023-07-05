"""
    plot(mf::AbstractMultiFlow, path::String)

A method for plotting multi flow instances
"""
function plot(mf::AbstractMultiFlow; 
              savepath::String="", 
              figsize::Tuple{Measures.Length,Measures.Length}=(16cm,16cm),
              nodelabel=AbstractArray=1:nv(mf.graph),
)
    plt=gplot(mf.graph,
        nodelabel=nodelabel,
    )
    if !isempty(savepath)
        draw(PNG(savepath, figsize[1], figsize[2]), plt)
    end
end

