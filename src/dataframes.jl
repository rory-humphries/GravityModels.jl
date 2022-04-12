function temporal_graph_from_df(df, time, source, destination, vals)
    vert_list = unique(union(df[:, source], df[:, destination]))
    vert_id_dict = Dict(zip(vert_list, 1:length(vert_list)))

    df_by_time = groupby(df, time)

    sp_mats = Vector{SparseMatrixCSC{Int,Int}}()

    for group in df_by_time
        push!(sp_mats, graph_from_df(group, source, destination, vals, vert_id_dict))
    end
    return sp_mats
end

function graph_from_df(df, source, destination, vals=nothing, vertex_id=nothing)
    vert_list = unique(union(df[:, source], df[:, destination]))
    if isnothing(vertex_id)
        v_map = Dict(zip(vert_list, 1:length(vert_list)))
    else
        v_map = vertex_id
    end
    num_verts = maximum(values(v_map))
    mat_size = (num_verts, num_verts)

    V = df[:, vals]
    I = [get(v_map, x, -1) for x in df[:, destination]]
    J = [get(v_map, x, -1) for x in df[:, source]]

    return sparse(I, J, V, mat_size...)
end

function count_occurances!(dfl, dfr, on; colname=:count)
    leftjoin!(dfl, combine(groupby(dfr, on[2]), nrow => colname); on=on)
    replace!(dfl[!, colname], missing => 0)
    disallowmissing!(dfl, colname)
    return dfl
end
