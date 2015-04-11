from nifty import *


class vector_field(field):
    
    def __init__(self,domain,Ndim=None,target=None,val=None,domain_explicit=False,**kwargs):
        if(domain_explicit):
            ## check domain
            if(not isinstance(domain,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            self.domain = domain
            ## check codomain
            if(target is None):
                target = domain.get_codomain()
            else:
                self.domain.check_codomain(target)
            self.target = target
        else:
            ## check domain
            if(not isinstance(domain,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            if(Ndim is None):
                Ndim = domain.naxes()
            self.domain = nested_space([point_space(Ndim),domain])
            ## check codomain
            if(target is None):
                target = domain.get_codomain()
            else:
                domain.check_codomain(target)
            self.target = nested_space([self.domain.nest[0],target])
        
        
        if(val is None):
            vals = self.domain.nest[1].get_random_values(codomain=self.target.nest[1],**kwargs)
            self.val = np.expand_dims(vals,0)
            for ii in range(1,Ndim):
                vals = self.domain.nest[1].get_random_values(codomain=self.target.nest[1],**kwargs)
                vals = np.expand_dims(vals,0)
                self.val = np.append(self.val,vals,axis=0)
        
        else: self.val = self.domain.enforce_values(val,extend=True)
        
    
    def transform(self,target=None,**kwargs):
        
        if(not(target is None)):
            target_domain = nested_space([self.target.nest[0],target])
        res = super(vector_field,self).transform(target=target,overwrite=False,**kwargs)
        
        return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
        
    
    def power(self,**kwargs):
        if("codomain" in kwargs):
            kwargs.__delitem__("codomain")
        pp = self.domain.nest[1].calc_power(self.val[0],codomain=self.target.nest[1],**kwargs)
        for ii in range(1,self.Ndim()):
            pp += self.domain.nest[1].calc_power(self.val[ii],codomain=self.target.nest[1],**kwargs)
        return pp
    
    def __pos__(self):
        return vector_field(self.domain,val=+self.val,target=self.target,domain_explicit=True)

    def __neg__(self):
        return vector_field(self.domain,val=-self.val,target=self.target,domain_explicit=True)

    def __abs__(self):
        if(np.iscomplexobj(self.val)):
            return np.absolute(self.val)
        else:
            return vector_field(self.domain,val=np.absolute(self.val),target=self.target,domain_explicit=True)
    
    
    def cross(self,x):
        
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return vector_field(x.domain,target=x.target,domain_explicit=True,val=np.cross(self.val.astype(x.domain.datatype),x.val,axis=0))
                else:
                    return vector_field(self.domain,target=self.target,domain_explicit=True,val=np.cross(self.val,x.val.astype(self.domain.datatype),axis=0))
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        
        else:
            x = self.domain.enforce_values(x,extend=False)
            return vector_field(self.domain,target=self.target,domain_explicit=True,val=np.cross(self.val,x,axis=0))
    
    def vec_dot(self,x):
        
        if(isinstance(x,field)):
            return field(self.domain.nest[1],target=self.target.nest[1],val=(self.__mul__(x)).val.sum(axis=0))
        elif(isinstance(x,np.ndarray)):
            if(x.shape == (self.Ndim(),)):
                res = self.component(0)*x[0]
                for ii in range(1,self.Ndim()):
                    res += self.component(ii)*x[ii]
                return res
            else:
                return field(self.domain.nest[1],target=self.target.nest[1],val=(self.__mul__(x)).val.sum(axis=0))
        else:
            return field(self.domain.nest[1],target=self.target.nest[1],val=(self.__mul__(x)).val.sum(axis=0))
            
    
    def vec_mul(self,x):
        if(isinstance(x,field)):
            if(self.domain.nest[0]==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    val = np.array([self.val.astype(x.domain.datatype)[iii]*x.val[iii] for iii in range(len(x.val))])
                    return vector_field(nested_space([x.domain,self.domain.nest[1]]),target=nested_space([x.domain.get_codomain(),self.target.nest[1]]),domain_explicit=True,val=val)
                else:
                    val = np.array([self.val[iii]*x.val.astype(self.domain.datatype)[iii] for iii in range(len(x.val))])
                    return vector_field(self.domain,target=self.target,domain_explicit=True,val=val)
            else:
                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        elif(isinstance(x,np.ndarray)):
            if(x.shape==(self.Ndim(),)):
                val = np.array([self.val[iii]*x.astype(self.domain.datatype)[iii] for iii in range(len(x))])
                return vector_field(self.domain,target=self.target,domain_explicit=True,val=val)
            else:
                return self.__mul__(x)
        else:
            return self.__mul__(x)
            
    
    def vec_div(self,x):
        if(isinstance(x,field)):
            if(self.domain.nest[0]==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    val = np.array([self.val.astype(x.domain.datatype)[iii]/x.val[iii] for iii in range(len(x.val))])
                    return vector_field(nested_space([x.domain,self.domain.nest[1]]),target=nested_space([x.domain.get_codomain(),self.target.nest[1]]),domain_explicit=True,val=val)
                else:
                    val = np.array([self.val[iii]/x.val.astype(self.domain.datatype)[iii] for iii in range(len(x.val))])
                    return vector_field(self.domain,target=self.target,domain_explicit=True,val=val)
            else:
                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        elif(isinstance(x,np.ndarray)):
            if(x.shape==(self.Ndim(),)):
                val = np.array([self.val[iii]/x.astype(self.domain.datatype)[iii] for iii in range(len(x))])
                return vector_field(self.domain,target=self.target,domain_explicit=True,val=val)
            else:
                return self.__div__(x)
        else:
            return self.__div__(x)
    
    
    def __mul__(self,x): ## __mul__ : self * x

        if(isinstance(x,field)):
            if(self.domain==x.domain):
                res = super(vector_field,self).__mul__(x)
                return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
            else:
                if(self.domain.nest[1]==x.domain):
                    if(x.domain.datatype>self.domain.datatype):
                        about.warnings.cprint("WARNING: codomain set to default.")
                        res_val = self.val.astype(x.domain.datatype)*x.val
                        res_domain = nested_space([self.domain.nest[0],x.domain])
                        res_target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                        return vector_field(res_domain,target=res_target,domain_explicit=True,val=res_val)
                    else:
                        res_val = self.val*x.val.astype(self.domain.datatype)
                        return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
                else:
                    raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        else:
            res_val = self.domain.enforce_values(x,extend=False)
            res_val = self.val*res_val
            return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)

    __rmul__ = __mul__  ## __rmul__ : x * self
    
    def __imul__(self,x): ## __imul__ : self *= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):

                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val *= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val *= x.val.astype(self.domain.datatype)
            elif(self.domain.nest[1]==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = nested_space([self.domain.nest[0],x.domain])
                    self.val *= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                else:
                    self.val *= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val *= x
        return self
    

    def __div__(self,x):

        if(isinstance(x,field)):
            if(self.domain==x.domain):
                res = super(vector_field,self).__div__(x)
                return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
            else:
                if(self.domain.nest[1]==x.domain):
                    if(x.domain.datatype>self.domain.datatype):
                        about.warnings.cprint("WARNING: codomain set to default.")
                        res_val = self.val.astype(x.domain.datatype)/x.val
                        res_domain = nested_space([self.domain.nest[0],x.domain])
                        res_target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                        return vector_field(res_domain,target=res_target,domain_explicit=True,val=res_val)
                    else:
                        res_val = self.val/x.val.astype(self.domain.datatype)
                        return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
                else:
                    raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        else:
            res_val = self.domain.enforce_values(x,extend=False)
            res_val = self.val/res_val
            return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
            
    __truediv__ = __div__

    def __rdiv__(self,x):

        if(isinstance(x,field)):
            if(self.domain==x.domain):
                res = super(vector_field,self).__rdiv__(x)
                return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
            else:
                if(self.domain.nest[1]==x.domain):
                    if(x.domain.datatype>self.domain.datatype):
                        about.warnings.cprint("WARNING: codomain set to default.")
                        res_val = x.val/self.val.astype(x.domain.datatype)
                        res_domain = nested_space([self.domain.nest[0],x.domain])
                        res_target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                        return vector_field(res_domain,target=res_target,domain_explicit=True,val=res_val)
                    else:
                        res_val = x.val.astype(self.domain.datatype)/self.val
                        return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
                else:
                    raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        else:
            res_val = self.domain.enforce_values(x,extend=False)
            res_val = res_val/self.val
            return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
    
    __rtruediv__ = __rdiv__
    
    def __idiv__(self,x): ## __idiv__ : self /= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):

                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val /= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val /= x.val.astype(self.domain.datatype)
            elif(self.domain.nest[1]==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = nested_space([self.domain.nest[0],x.domain])
                    self.val /= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                else:
                    self.val /= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val /= x
        return self

    
    def __add__(self,x):
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                res = super(vector_field,self).__add__(x)
                return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
            else:
                if(self.domain.nest[1]==x.domain):
                    if(x.domain.datatype>self.domain.datatype):
                        about.warnings.cprint("WARNING: codomain set to default.")
                        res_val = self.val.astype(x.domain.datatype)+x.val
                        res_domain = nested_space([self.domain.nest[0],x.domain])
                        res_target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                        return vector_field(res_domain,target=res_target,domain_explicit=True,val=res_val)
                    else:
                        res_val = self.val+x.val.astype(self.domain.datatype)
                        return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
                else:
                    raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        else:
            res_val = self.domain.enforce_values(x,extend=False)
            res_val = self.val+res_val
            return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
    
    __radd__ = __add__
    
    
    def __iadd__(self,x): ## __iadd__ : self += x
        if(isinstance(x,field)):
            if(self.domain==x.domain):

                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val += x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val += x.val.astype(self.domain.datatype)
            elif(self.domain.nest[1]==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = nested_space([self.domain.nest[0],x.domain])
                    self.val += x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                else:
                    self.val += x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val += x
        return self
        

    def __sub__(self,x):
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                res = super(vector_field,self).__sub__(x)
                return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
            else:
                if(self.domain.nest[1]==x.domain):
                    if(x.domain.datatype>self.domain.datatype):
                        about.warnings.cprint("WARNING: codomain set to default.")
                        res_val = self.val.astype(x.domain.datatype)-x.val
                        res_domain = nested_space([self.domain.nest[0],x.domain])
                        res_target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                        return vector_field(res_domain,target=res_target,domain_explicit=True,val=res_val)
                    else:
                        res_val = self.val-x.val.astype(self.domain.datatype)
                        return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
                else:
                    raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        else:
            res_val = self.domain.enforce_values(x,extend=False)
            res_val = self.val-res_val
            return vector_field(self.domain,target=self.target,domain_explicit=True,val=res_val)
            
    
    def __rsub__(self,x):
        return(-(self.__sub__(x)))
        
        
    def __isub__(self,x): ## __isub__ : self -= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):

                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val -= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val -= x.val.astype(self.domain.datatype)
            elif(self.domain.nest[1]==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = nested_space([self.domain.nest[0],x.domain])
                    self.val -= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = nested_space([self.domain.nest[0],x.domain.get_codomain()])
                else:
                    self.val -= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val -= x
        return self


    def __pow__(self,x):
        res = super(vector_field,self).__pow__(x)
        return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)
        
    def __rpow__(self,x):
        res = super(vector_field,self).__rpow__(x)
        return vector_field(res.domain,target=res.target,domain_explicit=True,val=res.val)

    
    def plot(self,**kwargs):
        
        if("title" in kwargs):
            title = kwargs.__getitem__("title") + " -- "
            kwargs.__delitem__("title")
        else:
            title = ""
        
        for ii in range(self.Ndim()):
            field(self.domain.nest[1],target=self.target.nest[1],val=self.val[ii]).plot(title=title+"component {}".format(ii),**kwargs)
        
    
    def Ndim(self):
        return self.domain.nest[0].para[0]
        
    def component(self,jjj):
        try:
            jj = int(jjj)
        except:
            raise TypeError("ERROR: invalid input.")
        
        if((jj<0) or (jj>(self.Ndim()-1))):
            raise ValueError("ERROR: index needs to be between 0 and {}".format(self.Ndim()-1))
        
        return field(self.domain.nest[1],target=self.target.nest[1],val=self.val[jj])
            


class position_field(vector_field):
    
    def __init__(self,domain,target=None,**kwargs):
        ## check domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = nested_space([point_space(domain.naxes()),domain])
        ## check codomain
        if(target is None):
            target = domain.get_codomain()
        else:
            domain.check_codomain(target)
        self.target = nested_space([self.domain.nest[0],target])
        
        ndim = domain.para[:domain.naxes()]
        
        temp_vecs = np.array(np.where(np.ones(ndim))).reshape(np.append(domain.naxes(),ndim))
        
        corr_vecs = domain.zerocenter()*ndim/2
        
        self.val = np.array([domain.vol[ii]*(temp_vecs[ii]-corr_vecs[ii]) for ii in range(len(domain.vol))])
        
        self.val = self.val[: :-1]