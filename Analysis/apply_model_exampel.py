
# cette fonction est la focntion original venant de vd.py dans la librairie du projet
# étudions ensemble son focntionnement qui est le centre du fonctionnement du model versatile
# pour rappel la focntion est appeleé dans la boucle DDIM a chaque étape afin de predire l'image précédente (donc l'image avec moins de bruit)
def apply_model(self, x_info, timesteps, c_info):
        # on récupérer les types des inputs 
        x_type, x = x_info['type'], x_info['x']
        c_type, c = c_info['type'], c_info['c']
        dtype = x.dtype

        hs = []

        from .openaimodel import timestep_embedding

        glayer_ptr = x_type if self.global_layer_ptr is None else self.global_layer_ptr
        model_channels = self.diffuser[glayer_ptr].model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).to(dtype)
        emb = self.diffuser[glayer_ptr].time_embed(t_emb)
        
        # création d'iterateur en focntion des entrés (images ou texte)
        d_iter = iter(self.diffuser[x_type].data_blocks)
        c_iter = iter(self.diffuser[c_type].context_blocks)

        i_order = self.diffuser[x_type].i_order
        m_order = self.diffuser[x_type].m_order
        o_order = self.diffuser[x_type].o_order

        # Ici on selectionner l'ordre des modules de diffusione t conditionnement 
        h = x
        for ltype in i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)

        for ltype in o_order:
            if ltype == 'load_hidden_feature':
                h = torch.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)
        o = h

        return o