function fit_gravity_dist_opt(GM, data, samples)
    function ret_func(p)
        gm = GM(p...)
        gd = GravityDistribution(gm, data)
        output = 0.0
        if gd.C == Inf
            output = Inf
        else
            output = -sum(log.([pdf(gd, x) for x in eachcol(samples)]))
        end
        return output
    end
    return ret_func
end

function fit_gravity_dist(gravity_model, node_df, trades_df, init_p)

    # remove trades with 0 distance
    delete!(trades_df, trades_df[:, :distance] .== 0)

    # need aggregated in and out degree of the herds for each trade
    transform!(groupby(trades_df, :move_from), nrow => :out_degree)
    transform!(groupby(trades_df, :move_to), nrow => :in_degree)

    count_occurances!(node_df, trades_df, :id => :move_from; colname=:out_degree)
    count_occurances!(node_df, trades_df, :id => :move_to; colname=:in_degree)

    # ignore any herds not involved in trading
    delete!(
        node_df,
        (node_df[!, :in_degree] .== 0) .& (node_df[!, :out_degree] .== 0),
    )

    D = FuncMatrix(copy(transpose([node_df.x node_df.y])), Euclidean())
    gd_data = GravityDistributionData(node_df.out_degree, node_df.in_degree, D)

    id_dict = Dict(y => x for (x, y) in enumerate(node_df.id))
    sample_data = [
        [id_dict[x] for x in trades_df.move_from]'
        [id_dict[x] for x in trades_df.move_to]'
    ]

    ########################################
    # fit paramters to model using MLE

    opt = Optim.optimize(
        fit_gravity_dist_opt(gravity_model, gd_data, sample_data),
        init_p;
        show_trace=true,
    )
    return Optim.minimizer(opt)
end
