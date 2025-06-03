1. Video:

Train:
python main_gesture.py 
(configuration: @hydra.main(version_base=None, config_path="conf/vid_config", config_name="vid_678"))

CANNOT QUANTIZE

inference:
python main_inference_avgresults (this is the one that runs inference + creates csv)
@hydra.main(version_base=None, config_path="conf/avgresult_config/vid", config_name="vid_inference_0129")

Prob of matching results: 0.907 for vid_inference_0129

WEIGHT MODEL:
/home/jchang13/fusion/mmegogesture-fusion/saved_model/0601_video



2. camonly

inference:
python main_inference_avgresults_quant2.py 
@hydra.main(version_base=None, config_path="conf/avgresult_config/quant2/camonly", config_name="camonly_inference_0129")

Prob of matching results: 0.78
* MAKE SURE THE WANDB IS TURNED TURN TO FALSE IN THE ORIGINAL YAML
* MAKE SURE QUANTIZED IS TURNED TO FALSE IF RUNNING FLOAT IN THE INFERENCE YAML

WEIGHT LOCATION:
path_model: '/home/jchang13/fusion/mmegogesture-fusion/saved_model/0528_cam/'

3. Feature level concat
This is run with the PHD's implementation (not the custom FusionClassifer/ FusionClassifierOptions)

inference:
python main_inference_avgresults.py
@hydra.main(version_base=None, config_path="conf/avgresult_config/concat", config_name="concat_inference_0129")

WEIGHT LOCATION:
path_model: '/home/jchang13/fusion/mmegogesture-fusion/saved_model/0512_att_vs_concat/concat_0129_last.pt'


4. Averaging Late Fusion (Float)
python main_inference_avgresults_quant2 
@hydra.main(version_base=None, config_path="conf/avgresult_config/quant/late", config_name="late_inference_0129")

*MAKE SURE WANDB ORIGINAL YAML FALSE
WEIGHT LOCATION:
'/home/jchang13/fusion/mmegogesture-fusion/saved_model/0525_late/'

5. Averaging Late Fusion (Quantized + Train 1 Epoch)
STEP 1:
quantize_finetune.py
@hydra.main(version_base=None, config_path="conf/quant2/late_config", config_name="late_345_finetune")

Then run 
python main_inference_avgresults_quant2.py
@hydra.main(version_base=None, config_path="conf/avgresult_config/quant/late", config_name="late_inference_0129")

* MAKE SURE PATH MODEL: '/home/jchang13/fusion/mmegogesture-fusion/saved_model/0530_late_finetune/'
* AND CSV_SAVE: '/home/jchang13/fusion/mmegogesture-fusion/vis_outputs/csv/quant/late_finetune/late_0129.csv'

6. Averaging Late Fusion (Quantized only)
STEP 1:
quantize_finetune.py
COMMENT OUT LINE 176-179 in this script

    # Fine tuning step
    # print("🔍 Running fine-tuning step...")
    # trainer.train()
    # wandb.finish()

Then run 
python main_inference_avgresults_quant2.py
@hydra.main(version_base=None, config_path="conf/avgresult_config/quant/late", config_name="late_inference_0129")

