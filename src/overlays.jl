@with_kw mutable struct MergingFeaturesOverlay <: SceneOverlay
    target_id::Int64 = 0
    color::Colorant = colorant"blue"
    line_width::Float64=0.5
end

@with_kw mutable struct MergingNeighborsOverlay <: SceneOverlay 
    target_id::Int64 = 0 
    env::MergingEnvironment = MergingEnvironment()
    color_fore::Colorant = colorant"blue"
    color_main::Colorant = colorant"green"
    line_width::Float64 = 0.5
    textparams::TextParams = TextParams(y_start=20, size=20)
end

function AutoViz.render!(rendermodel::RenderModel, overlay::MergingNeighborsOverlay,
                 scene::Scene, roadway::Roadway)
    textparams = overlay.textparams
    yₒ = textparams.y_start
    Δy = textparams.y_jump
    vehicle_index = findfirst(overlay.target_id, scene)
    veh_ego = scene[vehicle_index]
    fore, merge, fore_main, rear_main = get_neighbors(overlay.env, scene, overlay.target_id)
    if fore.ind != nothing 
        veh_oth = scene[fore.ind]
        A = get_front(veh_ego)
        B = get_rear(veh_oth)
        add_instruction!(rendermodel, render_line_segment, (A.x, A.y, B.x, B.y, overlay.color_fore, overlay.line_width))
        drawtext(@sprintf("d fore:   %10.3f", fore.Δs), yₒ + 0*Δy, rendermodel, textparams)
    end
    if fore_main.ind != nothing 
        veh_oth = scene[fore_main.ind]
        A = get_front(veh_ego)
        B = get_rear(veh_oth)
        add_instruction!(rendermodel, render_line_segment, (A.x, A.y, B.x, B.y, overlay.color_main, overlay.line_width))
        drawtext(@sprintf("d fore main: %10.3f", fore_main.Δs), yₒ + 1*Δy, rendermodel, textparams)
    end
    if rear_main.ind != nothing 
        veh_oth = scene[rear_main.ind]
        A = get_front(veh_ego)
        B = get_rear(veh_oth)
        add_instruction!(rendermodel, render_line_segment, (A.x, A.y, B.x, B.y, overlay.color_main, overlay.line_width))
        drawtext(@sprintf("d rear main:  %10.3f", rear_main.Δs), yₒ + 2*Δy, rendermodel, textparams)
    end 
    if merge.ind != nothing 
        veh_oth = scene[merge.ind]
        A = get_front(veh_ego)
        B = get_rear(veh_oth)
        add_instruction!(rendermodel, render_line_segment, (A.x, A.y, B.x, B.y, overlay.color_main, overlay.line_width))
        drawtext(@sprintf("d merge:  %10.3f", merge.Δs), yₒ + 3*Δy, rendermodel, textparams)
    end 
end


@with_kw mutable struct DistToMergeOverlay <: SceneOverlay
    target_id::Int64 = 0 
    env::MergingEnvironment = MergingEnvironment()
    color_point::Colorant = colorant"red"
    color_line::Colorant = colorant"black"
    point_radius::Float64 = 0.5
    line_width::Float64 = 0.5
    textparams::TextParams = TextParams(y_start=50)
end

function AutoViz.render!(rendermodel::RenderModel, overlay::DistToMergeOverlay,
                 scene::Scene, roadway::Roadway)
    # display merge point 
    mp = overlay.env.merge_point
    add_instruction!(rendermodel, render_circle, (mp.x, mp.y, overlay.point_radius, overlay.color_point))
    
    # display line to merge point + info
    vehind = findfirst(overlay.target_id, scene)
    if vehind != nothing 
        veh = scene[vehind]
        # plot line
        A = get_front(veh)
        # add_instruction!(rendermodel, render_dashed_line, ([A.x mp.x; A.y mp.y], overlay.color_line, overlay.line_width))
        dm = dist_to_merge(overlay.env, veh)
        ttm = time_to_merge(overlay.env, veh)
        disp_point = A + VecSE2(-10.0, 10.0, 0.0)
        if overlay.target_id == 1
            disp_point = A + VecSE2(-10.0, -10.0, 0.0)
        end
        text_overlay = TextOverlay(text=[@sprintf("dist=%3.2f m",dm),
                                         @sprintf("vel=%3.2f m/s", veh.state.v),
                                         @sprintf("ttm=%2.2f s", ttm)], 
                                   pos=disp_point,
                                   line_spacing = 0.15,
                                   font_size = 20,
                                   incameraframe=true)
        # render!(rendermodel, text_overlay, scene, roadway)
    end
end

@with_kw mutable struct MaskingOverlay <: SceneOverlay
    mdp::GenerativeMergingMDP = GenerativeMergingMDP()
    acc::Float64 = 0.0 # current acceleration 
    textparams::TextParams = TextParams(size=20, x = 300, y_start=110)
end

function AutoViz.render!(rendermodel::RenderModel, overlay::MaskingOverlay,
                 scene::Scene, roadway::Roadway)
        ss = AugScene(scene, (acc=overlay.acc,))
        acts = safe_actions(overlay.mdp, ss)
        textparams = overlay.textparams
        drawtext(@sprintf("Available Actions %s", acts), textparams.y_start, rendermodel, textparams)
        # drawtext(@sprintf("Actions %s", acts), textparams.y_start, rendermodel, textparams)

end
        
@with_kw mutable struct CooperativeIDMOverlay <: SceneOverlay
    model::CooperativeIDM = CooperativeIDM()
    targetid::Int64 = 0
    textparams::TextParams = TextParams(size = 20, x = 700, y_start=30)
end

function AutoViz.render!(rendermodel::RenderModel, overlay::CooperativeIDMOverlay,
    scene::Scene, roadway::Roadway)
    observe!(overlay.model, scene, roadway, overlay.targetid)
    veh = get_by_id(scene, overlay.targetid)
    mergeveh = get_by_id(scene, EGO_ID)
    textparams = overlay.textparams
    yₒ = textparams.y_start
    Δy = textparams.y_jump
    drawtext(@sprintf("Driver Model Info %1d:", overlay.targetid), yₒ, rendermodel, textparams)
    # drawtext(@sprintf("TTM threshold: %1.1f", overlay.model.ttm_threshold), yₒ + Δy, rendermodel, textparams)
    drawtext(@sprintf("cooperation level: %1.1f", overlay.model.c), yₒ + Δy, rendermodel, textparams)
    drawtext(@sprintf("act merge: %1.1f m/s^2", overlay.model.a_merge), yₒ + 2*Δy, rendermodel, textparams)
    drawtext(@sprintf("act idm:   %1.1f m/s^2", overlay.model.a_idm), yₒ + 3*Δy, rendermodel, textparams)
    drawtext(@sprintf("act final: %1.1f m/s^2", overlay.model.a), yₒ + 4*Δy, rendermodel, textparams)
    drawtext(@sprintf("dist at merge: %2.2f", overlay.model.dist_at_merge), yₒ + 5*Δy, rendermodel, textparams)
    drawtext(@sprintf("desired gap: %2.2f", overlay.model.s_des), yₒ + 6*Δy, rendermodel, textparams)
    ego_ttm = overlay.model.ego_ttm 
    veh_ttm = overlay.model.veh_ttm
    # @show !(ego_ttm < 0.0 || ego_ttm = Inf || ego_ttm <= veh_ttm || veh_ttm == Inf)
    # drawtext(@sprintf("considering merge: %s", !(ego_ttm < 0.0 || ego_ttm == Inf || ego_ttm < veh_ttm || veh_ttm == Inf || mergeveh.state.posF.s < overlay.model.fov)), yₒ + 7*Δy, rendermodel, textparams)
    drawtext(@sprintf("considering merge: %s", overlay.model.consider_merge), yₒ + 7*Δy, rendermodel, textparams)
    drawtext(@sprintf("driver ttm: %2.1f", overlay.model.ego_ttm), yₒ + 8*Δy, rendermodel, textparams)
    drawtext(@sprintf("merge ttm: %2.1f", overlay.model.veh_ttm), yₒ + 9*Δy, rendermodel, textparams)
    drawtext(@sprintf("front car %s", overlay.model.front_car), yₒ + 10*Δy, rendermodel, textparams)
end

function get_car_type_colors(scene::Scene, models::Dict{Int64, DriverModel};
                             cooperative::Colorant = RGBA(0.0, 1., 0., 1.),
                             aggressive::Colorant = RGBA(1.0,0.0,0.498, 1.),
                             cooperative_slow::Colorant = RGBA(0.8, 0.9, 0., 1.),
                             aggressive_slow::Colorant = RGBA(0.7,0.0,0.9, 1) )
    color_dict = Dict{Int64, Colorant}()
    for veh in scene
        if veh.id == 1
            color_dict[1] = COLOR_CAR_EGO
        elseif models[veh.id].c == 1 && models[veh.id].idm.v_des < 15.0
            color_dict[veh.id] = cooperative #cooperative_slow
        elseif models[veh.id].c == 0 && models[veh.id].idm.v_des < 15.0
            color_dict[veh.id] = aggressive # aggressive_slow
        elseif models[veh.id].c == 1
            color_dict[veh.id] = cooperative
        elseif models[veh.id].c == 0
            color_dict[veh.id] = aggressive
        end
    end   
    return color_dict
end

struct BeliefOverlay <: SceneOverlay
    b::MergingBelief
end

function AutoViz.render!(rendermodel::RenderModel, overlay::BeliefOverlay,
                        scene::Scene, roadway::Roadway)
    for (vehid,prob) in overlay.b.driver_types 
        vehind = findfirst(vehid, scene)
        if vehind != nothing 
            render_proba!(rendermodel, scene, roadway, vehind, prob)
        end
    end
end
     

function render_proba!(rendermodel::RenderModel, scene::Scene, roadway::Roadway, vehind::Int64, proba::Float64)
    veh = scene[vehind]
    A = get_front(veh)
    disp_point = A + VecSE2(-5.0, 4.0, 0.0)
    text_overlay = TextOverlay(text=[@sprintf("%1.2f", proba)], 
                                pos=disp_point,
                                line_spacing = 0.15,
                                font_size = 17,
                                incameraframe=true)
    render!(rendermodel, text_overlay, scene, roadway)
end

