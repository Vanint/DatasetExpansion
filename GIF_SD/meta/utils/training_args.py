import argparse

def create_argparser():
    parser = argparse.ArgumentParser(description='Diffuion model for Multimodel Image Generation')

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--glide_path', type = str,
                       help='path to your trained GLIDE, same as dalle_path')
    parser.add_argument('--dataset_name', type = str, default = 'tusou2E_clean',
                        help='name of dataset used to train')
    
    # model and diffusion | copy from def model_and_diffusion_defaults()
    parser.add_argument('--unclip', type = str, default = 'True',
                        help='for Dalle 2') 
    ## UNet00
    parser.add_argument('--image_size',  type =int,    default=64)
    parser.add_argument('--num_channels', type =int,   default=192)
    parser.add_argument('--num_res_blocks', type =int, default=3)
    parser.add_argument('--channel_mult', type =str,   default="")
    parser.add_argument('--num_heads', type =int,      default=1)
    parser.add_argument('--num_head_channels', type =int,  default=64)
    parser.add_argument('--num_heads_upsample', type =int, default=-1)
    parser.add_argument('--attention_resolutions', type =str, default="32,16,8")
    parser.add_argument('--dropout', type =float, default=0.1)
    parser.add_argument('--text_ctx', type =int, default=80, help="text max length | GLIDE 128")
    parser.add_argument('--max_text_len', type =int, default=80, help="maximum text length | DALLE")
    parser.add_argument('--cond_type', type=str, default='text', 
                                        help='condition type: text | none')
    ## transformer for text
    parser.add_argument('--xf_width', type =int, default=512)
    parser.add_argument('--xf_layers', type =int, default=16)
    parser.add_argument('--xf_heads', type =int, default=8)
    parser.add_argument('--xf_final_ln', type =bool, default=True)
    parser.add_argument('--xf_padding', type =bool, default=True)
    ## diffusion
    parser.add_argument('--empty_text_prob', type=float, default=0.5,
                    help="For cfg, probablity of replacing text to empty seq ")
    parser.add_argument('--empty_clip_prob', type=float, default=0.1,
                    help="For cfg, probablity of setting clip emb to 0")
    parser.add_argument('--empty_t5_prob', type=float, default=0.3,
                    help="For cfg, probablity of setting t5 emb to 0")

    parser.add_argument('--diffusion_steps', type =int, default=1000)
    parser.add_argument('--noise_schedule', type =str, default="cosine")
    parser.add_argument('--timestep_respacing', type =str, default="250", 
                                                help="# use 250 diffusion steps for fast sampling")
    parser.add_argument('--use_scale_shift_norm', type =bool, default=True)
    parser.add_argument('--resblock_updown', type =bool, default=True)
    parser.add_argument('--cache_text_emb', type =bool, default=False)
    parser.add_argument('--learn_sigma', action='store_true', default=False)
    
    ## aux
    parser.add_argument("--use_zero", action='store_true', default=False,
                         help="Using ZeroRedundancyOptimizer or not")
    parser.add_argument("--use_byteps", action='store_true', default=False,
                         help="Using ZeroRedundancyOptimizer or not")
    parser.add_argument('--use_fp16', action='store_true', default=False,
                        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--initial_lg_loss_scale', type=float, default=10.0, 
                        help="initial log loss scale for mix precision, found 20ls is a good initial for cc15m")
    parser.add_argument('--fp16_scale_growth', type=float, default=1e-3, 
                        help="increment of lg_loss_scale")
    parser.add_argument('--not_load_optimizer', action='store_true', default=False,
                        help='if true, will not load optimizer state_dict, only use when load fp32 model to continue train fp16 or vice versa')
    parser.add_argument('--inpaint', type =bool, default=False)
    parser.add_argument('--super_res', type =bool, default=False)

    ## byteps
    parser.add_argument('--staleness', type =int, default=0, 
                        help="Number of controleld gradients staleness for pipe-sgd. If set to 1,the parameter update is delayed by 1 step")
    parser.add_argument('--pipesgd_warmup_iter', type =int, default=0,
                        help="Number of warmup steps for pipe-sgd. during which pipsgd stalenesss is fixed at 0")
    # training 
    # aligned with glide
    parser.add_argument('--schedule_sampler', type = str, default = "uniform",
                        help='type of schedule_sampler')
    parser.add_argument('--train_batch_size', type = int, default = 6,
                        help='batch size for each gpu in distributed training')
    parser.add_argument('--learning_rate', type = float, default = 1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0,
                        help='weight decay (L2) regularization')
    parser.add_argument('--lr_anneal_steps', type = float, default = 0,
                        help='steps for learning rate annealing')
    parser.add_argument('--ema_rate', type = str, default = "0.9999",
                        help='rate for ema | comma-separated list of EMA values')
    # previous argument for DALLE
    parser.add_argument('--num_train_steps', type = int, default = 600000000,
                        help='number of train steps')
    parser.add_argument('--grad_clip_norm', type = float, default = 0.5,
                        help='grad_clip_norm')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for initialization')
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--logging_gen_steps", type=int, default=1000, help="Log Generated Image every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help=" 10000 Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=100,
                        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default")
    
    # visdom
    parser.add_argument('--VISDOM_ENV', type=str, default='chinese_fp32_v3', 
                        help='visdom env name')
    parser.add_argument('--VISDOM_PORT', type=int, default=8906, 
                        help='visdom port')
    parser.add_argument('--VISDOM_SERVER', type=str, default='http://10.227.88.58/', 
                        help='visdom ip address | yizhe http://10.227.75.143/')

    # text tokenizer type
    parser.add_argument('--tokenizer_type', type=str, default='bert', 
                        help='tokenizer type: bert | openai | hug')
    parser.add_argument('--lang_type', type=str, default='cn', 
                        help='languuage type: cn | en')

    # data path
    parser.add_argument('--train_data_path', type=str, default='hdfs://haruna/home/byte_labcv_gan/public/multimodal_all/tusou2E_clean',
                        help='path to your train data')
    parser.add_argument('--vocab_file', type=str, default='hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab',
                        help='path to your vocab file')
    parser.add_argument("--output_dir", type=str, default='hdfs://haruna/home/byte_labcv_gan/common/text2image/dalle_cn/checkpoints/chinese_fp32_v3',
                        help="The output directory where the model predictions and checkpoints will be written.")
    return parser
