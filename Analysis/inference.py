import os
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
from contextlib import nullcontext
import types

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
#from cusomized_gradio_blocks import create_myexamples, customized_as_example, customized_postprocess

import numpy as np
import torch
import torchvision.transforms as tvtrans
from PIL import Image
from IPython.display import display
from lib.model_zoo import get_model
from ddim import DDIMSampler
#from lib.model_zoo.ddim import DDIMSampler
from function import *
from PIL import Image
import torch



class VdInference:
    def __init__(self,ddim_steps=10, which='v1.0', fp16=True):
        ## fp16 est la version plus optimisé du modèle, le temps d'inference est plus rapide
        highlight_print(which)
        self.which = which
        # ici on vient charger le fichier yaml associé au modèle que l'on veut, ici on choisit notre modèle vd (virsatise diffusion)
        # CFG = Classifier Free Guidance, ça permet au modèle de gérer le systéme de guidance. On pourra lui passer notre text en input
        # et une chaine vide afin qu'il soit guidé ou non pour qu'il générer de la diversité 
        cfgm = model_cfg_bank()('vd_four_flow_v1-0')
        # net contient donc, le model pour encoder et decoder des images et du texte.
        # il contient également le model selectionné (ici VD  le modèle multi modal (texte image) Versatile diffusion (vd))
        # Enfin, dans ce model chargé, net a accés au model de prediction de bruit utilisé à chaque step dans DDIM
        net = get_model()(cfgm)

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("############## On utilise " + str(self.device) + " ############## ")

        # on selection GPU ou CPU
        self.dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32

        if fp16 and self.device == "cuda":
            highlight_print('Running in FP16 on GPU')
            net.ctx['text'].fp16 = True
            net.ctx['image'].fp16 = True
            net = net.half()  # on transforme les poids du modèle en float16 (souvent optimisé pour els GPU nvidia), on vient donc diviser en deux car par defaut on est en float32

        # On charge les poids pré entrainés du modèle associé (en 16 ou 32)
        sd_path = 'pretrained/vd-four-flow-v1-0-fp16.pth' if fp16 else 'pretrained/vd-four-flow-v1-0.pth'
        sd = torch.load(sd_path, map_location="cpu")
        net.load_state_dict(sd, strict=False)
        net.to(self.device)

        self.net = net
        # On donne tous les modèles et nottament le modèle vd qui contioent le model de prediction de bruit 
        self.sampler = DDIMSampler(net)  
        self.output_dim = [512, 512]
        self.n_sample_image = 1  
        self.ddim_steps = ddim_steps
        self.ddim_eta = 0.0
        self.scale_textto = 7.5
        self.image_latent_dim = 4
        self.text_latent_dim = 768
        self.n_sample_text = 4
        self.text_temperature = 1


        if which == 'v1.0':
            # adjust_rank permet de donner des informations sur le taux de variation de nos images lors de l'inference i2i
            self.adjust_rank_f = adjust_rank(max_drop_rank=[1, 5], q=20)
            self.scale_imgto = 7.5
            self.disentanglement_noglobal = True

    def inference_t2i(self, text, seed):
        n_samples = self.n_sample_image
        scale = self.scale_textto
        sampler = self.sampler
        h, w = self.output_dim

        # on vient encoder notre input
        # texte sans guidance, on donne une chaine vide, cela permet au modèle d'etre "créatif"
        u = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1).to(self.device)
        # texte avec guidance, on donne le texte qu'on veut en image
        c = self.net.ctx_encode([text], which='text').repeat(n_samples, 1, 1).to(self.device)

        # la taille de tenseur latent
        shape = [n_samples, self.image_latent_dim, h//8, w//8]

        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed + 100)

        # ici on vient applique notre modèle de diffusion qui va générer notre image
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type': 'image'},
            c_info={'type': 'text', 'conditioning': c, 'unconditional_conditioning': u,
                    'unconditional_guidance_scale': scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta
        )

        # on décode notre resultat généré x pour qu'il resorte non pas en vecteur latent mais en image
        im = self.net.vae_decode(x.to(self.device), which='image')
        im = [tvtrans.ToPILImage()(i.cpu().float()) for i in im] 

        return im 
    
    # on prend une image en entré, on l'encode en espace latent, on modifie plus ou moins l'espace à l'aide du modèle de diffusion et on décode en image
    def inference_i2i(self, im, fid_lvl, fcs_lvl, clr_adj, seed):
        n_samples = self.n_sample_image
        scale = self.scale_imgto
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        BICUBIC = PIL.Image.Resampling.BICUBIC
        im = im.resize([w, h], resample=BICUBIC)

        # si fid_lvl est égale à 1 alors cela signifi qu'il n'y a pas de modification de notre image. On reste comme en input sans passer par le modèle de diffusion
        # 1 est la valeur maximum et 0 la valeur minimu qui veut dire " beaucoup de changement par rapport à l'image de base"
        if fid_lvl == 1:
            return [im]*n_samples

        cx = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)

        # on encode notre image d'input avec l'encodeur de net pour extraire le contexte
        c = self.net.ctx_encode(cx, which='image')

        # cette partie permet de créer c qui sera le contexte de notre image, modifié par notre fonction. Ce contexte modifié sera donné en input avec l'image
        if self.disentanglement_noglobal:
            # on sépare les informations globales et locales de notre vecteur encodé. 
            c_glb = c[:, 0:1]
            c_loc = c[:, 1: ]
            # on vient donner les information à notre fonction adjust_ranf_f afin qu'il modifie une partie de nos vecteurs locaux. 
            # Il modifie en fonction de fcs_lvl, soit le lvl de modification de notre image
            c_loc = self.adjust_rank_f(c_loc, fcs_lvl)
            # on merge le vecteur global et les locaux
            c = torch.cat([c_glb, c_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            c = self.adjust_rank_f(c, fcs_lvl).repeat(n_samples, 1, 1)

        # vecteur non guidé (équivalent à une image bruité en input)
        u = torch.zeros_like(c)

        # la shape de notre vecteur latent, pour etre donné au DDIM
        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        # dans le cas ou l'image est modifié mais pas au point d'etre un bruit (0)
        if fid_lvl!=0:
            # on encode notre image pour avoir le latent pour DDIM
            x0 = self.net.vae_encode(cx, which='image').repeat(n_samples, 1, 1, 1)
            # le nombre de step depend du tax de modification donné par fid_lvl. Si fid_lvl est petit alors il y aura plus de modification par rapport à l'image input
            # alors il y aura plus de steps !
            step = int(self.ddim_steps * (1-fid_lvl))

            # ici on vient applique notre modèle de diffusion qui va générer notre image en donner le context en parametres
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)
        else:
            # dans le cas où fid_lvl alors on a une image qui n'a plus rien a voir avec celle d'entré. Donc on ne lui donne pas l'image pour qu'il générer une image depuis un bruit.
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image',},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)
        # on decode le resultat de notre modèle de diffusion
        imout = self.net.vae_decode(x, which='image')

        # on garde ls informations de colorimétrie de notre image de base
        if clr_adj == 'Simple':
            cx_mean = cx.view(3, -1).mean(-1)[:, None, None]
            cx_std  = cx.view(3, -1).std(-1)[:, None, None]
            imout_mean = [imouti.view(3, -1).mean(-1)[:, None, None] for imouti in imout]
            imout_std  = [imouti.view(3, -1).std(-1)[:, None, None] for imouti in imout]
            imout = [(ii-mi)/si*cx_std+cx_mean for ii, mi, si in zip(imout, imout_mean, imout_std)]
            imout = [torch.clamp(ii, 0, 1) for ii in imout]

        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout

    
    def inference_i2t(self, im, seed):
        n_samples = self.n_sample_text
        scale = self.scale_imgto
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        BICUBIC = PIL.Image.Resampling.BICUBIC
        im = im.resize([w, h], resample=BICUBIC)

        cx = tvtrans.ToTensor()(im)[None].to(device)
        # on vient encoder avec et sans guidance donc avec l'image d'entré et sans l'image d'entré (une image noir plus precisement )
        c = self.net.ctx_encode(cx, which='image').repeat(n_samples, 1, 1)
        u = self.net.ctx_encode(torch.zeros_like(cx), which='image').repeat(n_samples, 1, 1)
        # shape du vecteir pour DDIM
        shape = [n_samples, self.text_latent_dim]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        # on donne au model de diffusion
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'text',},
            c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)
        # on decode le vecteur en texte
        tx = self.net.vae_decode(x, which='text', temperature=self.text_temperature)
        # onb enlever les doublons
        tx = [remove_duplicate_word(txi) for txi in tx]
        tx_combined = '\n'.join(tx)
        return tx_combined

    # c'est sensiblement la même chose que pour i2t mais cette fois ci on a un texte en entré qui est encodé
    def inference_t2t(self, text, seed):
        n_samples = self.n_sample_text
        scale = self.scale_textto
        sampler = self.sampler
        u = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
        c = self.net.ctx_encode([text], which='text').repeat(n_samples, 1, 1)
        shape = [n_samples, self.text_latent_dim]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'text',},
            c_info={'type':'text', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)
        tx = self.net.vae_decode(x, which='text', temperature=self.text_temperature)
        tx = [remove_duplicate_word(txi) for txi in tx]
        tx_combined = '\n'.join(tx)
        return tx_combined
