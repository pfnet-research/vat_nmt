# Virtual Adversarial Training for NMT (Transformer model)
Implementation of "Effective Adversarial Regularization for Neural Machine Translation", ACL 2019

## References
Motoki Sato, Jun Suzuki, Shun Kiyono. "Effective Adversarial Regularization for Neural Machine Translation", ACL 2019
[URL]

# How to use
## Requirements

- Python3.6+
- Chainer 6.x+
- Cupy 6.x+
```
# install chainer and cupy
$ pip install cupy
$ pip install chainer
$ pip install logzero
```
Please see how to install chainer: https://docs.chainer.org/en/stable/install.html

## Train (iwslt2016-de-en)
```
$ python3 -u chainer_transformer.py --mode train --gpus 0 --dataset iwslt2016-de-en --seed 1212 --epoch 40 --out model_transformer_de-en
```

## Train with VAT (iwslt2016-de-en)
```
$ python3 -u chainer_transformer.py --mode train --gpus 0 --dataset iwslt2016-de-en --seed 1212 --epoch 40 --out model_transformer_de-en_vat_enc --use-vat 1 --eps 1.0 --perturbation-target 0
```

### perturbation types

| perturbation-target | (enc, dec, enc-dec) |
----|----
| 0 | enc |
| 1 | dec |
| 0 1 | enc-dec (both) |


### VAT, Adv, VAT-Adv
| use-vat | (vat, adv, vat-adv) |
----|----
| 0 | non (baseline) |
| 1 | vat |
| 2 | adv |
| 3 | vat-adv (both) |


## Eval
```
$ python3 -u chainer_transformer.py --mode test --gpus 0 --dataset iwslt2016-de-en --batchsize 600 --model model_transformer_de-en/model_epoch_40.npz --beam 20 --max-length 60 --datatype eval1
```

# License
MIT License. Please see the LICENSE file for details.
