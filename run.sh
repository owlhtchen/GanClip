python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=.\pretrained\ffhq.pkl
python generate.py --outdir=out --trunc=0.7 --seeds=600-605    --network=.\pretrained\metfaces.pkl
python generate.py --outdir=out --seeds=0-35 --class=1 --network=./pretrained/cifar10.pkl
python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293   --network=./pretrained/metfaces.pkl
python projector.py --outdir=out_project --target=./outffhq/seed0085.png --network=./pretrained/ffhq.pkl