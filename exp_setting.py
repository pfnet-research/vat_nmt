

def set_exp_dataset(exp_dict, args):
    args.encVocabFile = exp_dict['enc_vocab']
    args.decVocabFile = exp_dict['dec_vocab']
    args.encDataFile = exp_dict['enc']
    args.decDataFile = exp_dict['dec']
    args.encDevelDataFile = exp_dict['enc_dev']
    args.decDevelDataFile = exp_dict['dec_dev']
    if args.semi:
        if args.use_cheat:
            args.encDataFileSemi = exp_dict['semi_enc']
            args.decDataFileSemi = exp_dict['semi_dec']
        elif args.use_predict_decside:
            args.encDataFileSemi = exp_dict['semi_enc']
            args.decDataFileSemi = exp_dict['semi_dec_predict']
        else:
            args.encDataFileSemi = exp_dict['semi_enc_predict']
            args.decDataFileSemi = exp_dict['semi_dec']



def get_exp_dataset(name, basedir='dataset/'):
    exp_dict = {}
    if name == 'iwslt2016-de-en':
        exp_dict['enc_vocab'] = basedir + 'iwslt2016/de-en/train.de_bpe16000.vocab'
        exp_dict['dec_vocab'] = basedir + 'iwslt2016/de-en/train.en_bpe16000.vocab'
        exp_dict['enc'] = basedir + 'iwslt2016/de-en/train.de_bpe16000'
        exp_dict['dec'] = basedir + 'iwslt2016/de-en/train.en_bpe16000'
        exp_dict['enc_dev'] = basedir + 'iwslt2016/de-en/tst2012.de_bpe16000'
        exp_dict['dec_dev'] = basedir + 'iwslt2016/de-en/tst2012.en_bpe16000'
        exp_dict['enc_eval_1'] = basedir + 'iwslt2016/de-en/tst2013.de_bpe16000'
        exp_dict['dec_eval_1'] = basedir + 'iwslt2016/de-en/tst2013.en_bpe16000'
        exp_dict['enc_eval_2'] = basedir + 'iwslt2016/de-en/tst2014.de_bpe16000'
        exp_dict['dec_eval_2'] = basedir + 'iwslt2016/de-en/tst2014.en_bpe16000'
        # exp_dict['semi_enc'] = basedir + 'wmt16/train.tok.clean.iwslt.16000.de'
        # exp_dict['semi_dec'] = basedir + 'wmt16/train.tok.clean.iwslt.16000.en'
        exp_dict['semi_enc'] = basedir + 'iwslt_wmt_semi/wmt16_train.tok.clean.iwslt.16000.de'
        exp_dict['semi_dec'] = basedir + 'iwslt_wmt_semi/wmt16_train.tok.clean.iwslt.16000.en'
        # exp_dict['semi_enc_predict'] = basedir + 'wmt16_predict/predict.iwslt.16000.de'
        # exp_dict['semi_dec_predict'] = basedir + 'wmt16_predict/predict.iwslt.16000.en'
        exp_dict['semi_enc_predict'] = basedir + 'iwslt_wmt_semi/de-en.en.train1000000.txt'
        exp_dict['semi_dec_predict'] = basedir + 'iwslt_wmt_semi/de-en.de.train1000000.txt'

    elif name == 'iwslt2016-en-de':
        exp_dict['enc_vocab'] = basedir + 'iwslt2016/de-en/train.en_bpe16000.vocab'
        exp_dict['dec_vocab'] = basedir + 'iwslt2016/de-en/train.de_bpe16000.vocab'
        exp_dict['enc'] = basedir + 'iwslt2016/de-en/train.en_bpe16000'
        exp_dict['dec'] = basedir + 'iwslt2016/de-en/train.de_bpe16000'
        exp_dict['enc_dev'] = basedir + 'iwslt2016/de-en/tst2012.en_bpe16000'
        exp_dict['dec_dev'] = basedir + 'iwslt2016/de-en/tst2012.de_bpe16000'
        exp_dict['enc_eval_1'] = basedir + 'iwslt2016/de-en/tst2013.en_bpe16000'
        exp_dict['dec_eval_1'] = basedir + 'iwslt2016/de-en/tst2013.de_bpe16000'
        exp_dict['enc_eval_2'] = basedir + 'iwslt2016/de-en/tst2014.en_bpe16000'
        exp_dict['dec_eval_2'] = basedir + 'iwslt2016/de-en/tst2014.de_bpe16000'
        # exp_dict['semi_enc'] = basedir + 'wmt16/train.tok.clean.iwslt.16000.en'
        # exp_dict['semi_dec'] = basedir + 'wmt16/train.tok.clean.iwslt.16000.de'
        # exp_dict['semi_enc_predict'] = basedir + 'wmt16_predict/predict.iwslt.16000.en'
        # exp_dict['semi_dec_predict'] = basedir + 'wmt16_predict/predict.iwslt.16000.de'
        exp_dict['semi_enc'] = basedir + 'iwslt_wmt_semi/wmt16_train.tok.clean.iwslt.16000.en'
        exp_dict['semi_dec'] = basedir + 'iwslt_wmt_semi/wmt16_train.tok.clean.iwslt.16000.de'
        exp_dict['semi_enc_predict'] = basedir + 'iwslt_wmt_semi/de-en.de.train1000000.txt'
        exp_dict['semi_dec_predict'] = basedir + 'iwslt_wmt_semi/de-en.en.train1000000.txt'
    elif name == 'iwslt2016-fr-en':
        exp_dict['enc_vocab'] = basedir + 'iwslt2016/fr-en/train.fr_bpe16000.vocab'
        exp_dict['dec_vocab'] = basedir + 'iwslt2016/fr-en/train.en_bpe16000.vocab'
        exp_dict['enc'] = basedir + 'iwslt2016/fr-en/train.fr_bpe16000'
        exp_dict['dec'] = basedir + 'iwslt2016/fr-en/train.en_bpe16000'
        exp_dict['enc_dev'] = basedir + 'iwslt2016/fr-en/tst2012.fr_bpe16000'
        exp_dict['dec_dev'] = basedir + 'iwslt2016/fr-en/tst2012.en_bpe16000'
        exp_dict['enc_eval_1'] = basedir + 'iwslt2016/fr-en/tst2013.fr_bpe16000'
        exp_dict['dec_eval_1'] = basedir + 'iwslt2016/fr-en/tst2013.en_bpe16000'
        exp_dict['enc_eval_2'] = basedir + 'iwslt2016/fr-en/tst2014.fr_bpe16000'
        exp_dict['dec_eval_2'] = basedir + 'iwslt2016/fr-en/tst2014.en_bpe16000'
        exp_dict['semi_enc'] = basedir + 'iwslt_wmt_semi/wmt14_predict_en_fr.iwslt.16000.fr'
        exp_dict['semi_dec'] = basedir + 'iwslt_wmt_semi/wmt14_predict_en_fr.iwslt.16000.en'
        exp_dict['semi_enc_predict'] = basedir + 'iwslt_wmt_semi/fr-en.en.train900000.txt'
        exp_dict['semi_dec_predict'] = basedir + 'iwslt_wmt_semi/fr-en.fr.train900000.txt'
    elif name == 'iwslt2016-en-fr':
        exp_dict['enc_vocab'] = basedir + 'iwslt2016/fr-en/train.en_bpe16000.vocab'
        exp_dict['dec_vocab'] = basedir + 'iwslt2016/fr-en/train.fr_bpe16000.vocab'
        exp_dict['enc'] = basedir + 'iwslt2016/fr-en/train.en_bpe16000'
        exp_dict['dec'] = basedir + 'iwslt2016/fr-en/train.fr_bpe16000'
        exp_dict['enc_dev'] = basedir + 'iwslt2016/fr-en/tst2012.en_bpe16000'
        exp_dict['dec_dev'] = basedir + 'iwslt2016/fr-en/tst2012.fr_bpe16000'
        exp_dict['enc_eval_1'] = basedir + 'iwslt2016/fr-en/tst2013.en_bpe16000'
        exp_dict['dec_eval_1'] = basedir + 'iwslt2016/fr-en/tst2013.fr_bpe16000'
        exp_dict['enc_eval_2'] = basedir + 'iwslt2016/fr-en/tst2014.en_bpe16000'
        exp_dict['dec_eval_2'] = basedir + 'iwslt2016/fr-en/tst2014.fr_bpe16000'
        exp_dict['semi_enc'] = basedir + 'iwslt_wmt_semi/wmt14_predict_en_fr.iwslt.16000.en'
        exp_dict['semi_dec'] = basedir + 'iwslt_wmt_semi/wmt14_predict_en_fr.iwslt.16000.fr'
        exp_dict['semi_enc_predict'] = basedir + 'iwslt_wmt_semi/fr-en.fr.train900000.txt'
        exp_dict['semi_dec_predict'] = basedir + 'iwslt_wmt_semi/fr-en.en.train900000.txt'
    elif name == 'wmt16-en-de-fairseq':
        exp_dict['joint_vocab'] = basedir + 'wmt16_en_de_bpe32k/vocab.bpe.32000'
        exp_dict['enc_vocab'] = basedir + 'wmt16_en_de_bpe32k/vocab.bpe.32000.en'
        exp_dict['dec_vocab'] = basedir + 'wmt16_en_de_bpe32k/vocab.bpe.32000.de'
        exp_dict['enc'] = basedir + 'wmt16_en_de_bpe32k/train.tok.clean.bpe.32000.en'
        exp_dict['dec'] = basedir + 'wmt16_en_de_bpe32k/train.tok.clean.bpe.32000.de'
        exp_dict['enc_dev'] = basedir + 'wmt16_en_de_bpe32k/newstest2013.tok.bpe.32000.en'
        exp_dict['dec_dev'] = basedir + 'wmt16_en_de_bpe32k/newstest2013.tok.bpe.32000.de'
        exp_dict['enc_eval_1'] = basedir + 'wmt16_en_de_bpe32k/newstest2014.tok.bpe.32000.en'
        exp_dict['dec_eval_1'] = basedir + 'wmt16_en_de_bpe32k/newstest2014.tok.bpe.32000.de'

    elif name == 'wmt16-en-de':
        exp_dict['joint_vocab'] = basedir + 'wmt16/vocab.bpe.32000'
        exp_dict['enc_vocab'] = basedir + 'wmt16/vocab.bpe.32000.en'
        exp_dict['dec_vocab'] = basedir + 'wmt16/vocab.bpe.32000.de'
        exp_dict['enc'] = basedir + 'wmt16/train.tok.clean.bpe.32000.en'
        exp_dict['dec'] = basedir + 'wmt16/train.tok.clean.bpe.32000.de'
        exp_dict['enc_dev'] = basedir + 'wmt16/newstest2010.tok.bpe.32000.en'
        exp_dict['dec_dev'] = basedir + 'wmt16/newstest2010.tok.bpe.32000.de'
        exp_dict['enc_eval_1'] = basedir + 'wmt16/newstest2016.tok.bpe.32000.en'
        exp_dict['dec_eval_1'] = basedir + 'wmt16/newstest2016.tok.bpe.32000.de'
        exp_dict['enc_eval_2'] = basedir + 'wmt16/newstest2015.tok.bpe.32000.en'
        exp_dict['dec_eval_2'] = basedir + 'wmt16/newstest2015.tok.bpe.32000.de'
        exp_dict['enc_eval_3'] = basedir + 'wmt16/newstest2014.tok.bpe.32000.en'
        exp_dict['dec_eval_3'] = basedir + 'wmt16/newstest2014.tok.bpe.32000.de'
    elif name == 'wmt16-de-en':
        exp_dict['joint_vocab'] = basedir + 'wmt16/vocab.bpe.32000'
        exp_dict['enc_vocab'] = basedir + 'wmt16/vocab.bpe.32000.de'
        exp_dict['dec_vocab'] = basedir + 'wmt16/vocab.bpe.32000.en'
        exp_dict['enc'] = basedir + 'wmt16/train.tok.clean.bpe.32000.de'
        exp_dict['dec'] = basedir + 'wmt16/train.tok.clean.bpe.32000.en'
        exp_dict['enc_dev'] = basedir + 'wmt16/newstest2013.tok.bpe.32000.de'
        exp_dict['dec_dev'] = basedir + 'wmt16/newstest2013.tok.bpe.32000.en'
        exp_dict['enc_eval_1'] = basedir + 'wmt16/newstest2016.tok.bpe.32000.de'
        exp_dict['dec_eval_1'] = basedir + 'wmt16/newstest2016.tok.bpe.32000.en'
        exp_dict['enc_eval_2'] = basedir + 'wmt16/newstest2015.tok.bpe.32000.de'
        exp_dict['dec_eval_2'] = basedir + 'wmt16/newstest2015.tok.bpe.32000.en'
        exp_dict['enc_eval_3'] = basedir + 'wmt16/newstest2014.tok.bpe.32000.de'
        exp_dict['dec_eval_3'] = basedir + 'wmt16/newstest2014.tok.bpe.32000.en'
    elif name == 'wmt14-en-de':
        exp_dict['joint_vocab'] = basedir + 'wmt14_fairseq_vocab/wmt14_en_de/vocab_format.all'
        exp_dict['enc_vocab'] = basedir + 'wmt14_fairseq/wmt14_en_de/vocab_format.en'
        exp_dict['dec_vocab'] = basedir + 'wmt14_fairseq/wmt14_en_de/vocab_format.de'
        exp_dict['enc'] = basedir + 'wmt14_fairseq/wmt14_en_de/train.en'
        exp_dict['dec'] = basedir + 'wmt14_fairseq/wmt14_en_de/train.de'
        exp_dict['enc_dev'] = basedir + 'wmt14_fairseq/wmt14_en_de/valid.en'
        exp_dict['dec_dev'] = basedir + 'wmt14_fairseq/wmt14_en_de/valid.de'
        exp_dict['enc_eval_1'] = basedir + 'wmt14_fairseq/wmt14_en_de/test.en'
        exp_dict['dec_eval_1'] = basedir + 'wmt14_fairseq/wmt14_en_de/test.de'

    elif name == 'wmt14-en-de-nfkc':
        exp_dict['joint_vocab'] = basedir + 'wmt14/train.en-de.norm.codes'
        exp_dict['enc_vocab'] = basedir + 'wmt14/train.en-de.norm.codes'
        exp_dict['dec_vocab'] = basedir + 'wmt14/train.en-de.norm.codes'
        exp_dict['enc'] = basedir + 'wmt14/train.en.norm.input.nodev'
        exp_dict['dec'] = basedir + 'wmt14/train.de.norm.input.nodev'
        exp_dict['enc_dev'] = basedir + 'wmt14/valid.en.norm.input'
        exp_dict['dec_dev'] = basedir + 'wmt14/valid.de.norm.input'
        exp_dict['enc_eval_1'] = basedir + 'wmt14/newstest2014.en.norm.input'
        exp_dict['dec_eval_1'] = basedir + 'wmt14/newstest2014.de.norm.input'

    elif name == 'wmt14-en-de-sentencepiece':
        exp_dict['joint_vocab'] = basedir + 'open_nmt/wmtende.vocab'
        exp_dict['enc_vocab'] = basedir + 'open_nmt/wmtende.vocab'
        exp_dict['dec_vocab'] = basedir + 'open_nmt/wmtende.vocab'
        exp_dict['enc'] = basedir + 'open_nmt/train.en'
        exp_dict['dec'] = basedir + 'open_nmt/train.de'
        exp_dict['enc_dev'] = basedir + 'open_nmt/valid.en'
        exp_dict['dec_dev'] = basedir + 'open_nmt/valid.de'
        exp_dict['enc_eval_1'] = basedir + 'open_nmt/test.en'
        exp_dict['dec_eval_1'] = basedir + 'open_nmt/test.de'
    elif name == 'wmt14-en-fr':
        exp_dict['enc_vocab'] = basedir + 'wmt14_fairseq/wmt14_en_fr/vocab_format.en'
        exp_dict['dec_vocab'] = basedir + 'wmt14_fairseq/wmt14_en_fr/vocab_format.fr'
        exp_dict['enc'] = basedir + 'wmt14_fairseq/wmt14_en_fr/train.en'
        exp_dict['dec'] = basedir + 'wmt14_fairseq/wmt14_en_fr/train.fr'
        exp_dict['enc_dev'] = basedir + 'wmt14_fairseq/wmt14_en_fr/valid.en'
        exp_dict['dec_dev'] = basedir + 'wmt14_fairseq/wmt14_en_fr/valid.fr'
        exp_dict['enc_eval_1'] = basedir + 'wmt14_fairseq/wmt14_en_fr/test.en'
        exp_dict['dec_eval_1'] = basedir + 'wmt14_fairseq/wmt14_en_fr/test.fr'

    # TODO: WMT'16 14 En-De

    return exp_dict
