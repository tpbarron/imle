import im_expl

def bnn_process_f(args, replay_memory, dynamics, actor_critic):
    """
    A process that access the replay through shared memory and continuously
    updates the bnn in the background
    """
    while True:
        # print ("BNN proc: ", replay_memory.size)
        if (args.imle or args.vime) and replay_memory.size >= args.min_replay_size:
            print ("Updating BNN")
            # obs_mean, obs_std, act_mean, act_std = memory.mean_obs_act()
            _inputss, _targetss, _actionss = [], [], []
            for _ in range(args.bnn_n_updates_per_step):
                batch = replay_memory.sample(args.bnn_batch_size)
                obs_data = batch['observations'] #(batch['observations'] - obs_mean) / (obs_std + 1e-8)
                next_obs_data = batch['next_observations'] #(batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
                act_data = batch['actions'] #(batch['actions'] - act_mean) / (act_std + 1e-8)

                _inputss.append(obs_data)
                _targetss.append(next_obs_data)
                _actionss.append(act_data)

            # update bnn
            if args.vime:
                pre_bnn_error, post_bnn_error = im_expl.vime_bnn_update(dynamics, _inputss, _actionss, _targetss)
            elif args.imle:
                pre_bnn_error, post_bnn_error = im_expl.imle_bnn_update(actor_critic, dynamics, _inputss, _actionss, _targetss, use_cuda=args.cuda)
