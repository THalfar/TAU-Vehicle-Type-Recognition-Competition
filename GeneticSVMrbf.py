import random
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

"""
Nämä ovat evoluutio vakioita. Muuta tarpeen mukaan
"""
GEENISIIRTO = 0.2
GEENIALUSTUS = 0.5
YHDENMUTRATE = 0.08
INVAASIORAJA = 0.0025

"""
Nämä ovat geenin muoto
"""
MUUT = 1
GAMMA_INT = 2
GAMMA_FLOAT = 13
C_INT = 2
C_FLOAT = 13

"""
Muutta binit floateiksi katkaisukohtien mukaan.
Poistaa nollatapaukset korvaamalla ne ykkösellä.
"""
def binFloatiksi(kromosomi, alkuIso, loppuIso, loppuPieni):
    
    isot = 0.0    
    
    isotKerroin = binIntiksi(kromosomi, alkuIso, loppuIso)
    
    isot = 2**isotKerroin
                  
    pienet = 0.0    
    for i in range(loppuIso,loppuPieni):        
        pienet = pienet + kromosomi[i]*2.0**(loppuIso-1-i)
    
   
    if pienet*isot == 0.0:
        
        if isot == 0 and pienet > 0:            
            return pienet
        
        else:
            return 1         
        
    return isot*pienet


"""
Muuttaa binit intiksi katkaisukohtien mukaan
"""
def binIntiksi(kromosomi, alkuKohta, loppuKohta):
    
    ulos = 0        
    for i in range(alkuKohta, loppuKohta):
        ulos = ulos + kromosomi[i]*2**(i-alkuKohta)            
 
    return ulos


class GeneticSVMrbf():
    
    def __init__(self, n_gen, size, n_best, n_rand, 
                 n_children, mutation_rate, verbose = 0, tulostetaanKehitys = False,
                 tulostetaanAikakausi = False, aloituskerroin = 1, cv = 5, cvid = False,
                 eliittikerroin = [], luokkapainot = False):
        
        # Painotetaanko opetus luokkien mukaan
        self.luokkapainot = luokkapainot
        
        # Eliitti lasten määrät listana
        self.eliittikerroin = eliittikerroin
        
        # Tasapainotetaanko luokkien esiintymisen mukaan vai ei CV pisteet
        self.cvid = cvid
        
        #SearchCV cv arvo
        self.cv = cv
        
        # Mikä kerroin aloituspopulaatiolle
        self.aloituskerroin = aloituskerroin
        
        # Tulostetaanko aikakaudet
        self.tulostetaanAikakausi = tulostetaanAikakausi
        
        # Tulostetaanko kehitys lopuksi
        self.tulostetaanKehitys = tulostetaanKehitys
        
        # Tulostuksen määrä        
        self.verbose = verbose
        
        # Parhaat kromosomit per maailma
        self.kromosomis_best = []
        
        self.parhaat = []
        # Aikakausien määrä
        self.n_gen = n_gen
        # Yksilöiden määrä
        self.size = size
        # Montako parasta uuteen sukupolveen
        self.n_best = n_best
        # Montako satunnaista uuteen sukupolveen
        self.n_rand = n_rand
        # Montako lasta per uuden sukupolven yksilö
        self.n_children = n_children
        # Mutaation todennäköisyys per yksilö
        self.mutation_rate = mutation_rate
        
        self.generation = 0
        
        self.parasKromosomi = 0
        self.paras = 0
        
       
        if int(self.n_best + self.n_rand) * self.n_children != self.size:
            raise ValueError("The population size is not stable.")  
           
            
    def initilize(self):
        
        population = []
        
        for i in range(self.size*self.aloituskerroin):
            
            kromosomi = np.zeros(self.n_genes, dtype=np.int)            
            mask = np.random.rand(len(kromosomi)) < GEENIALUSTUS
            kromosomi[mask] = 1
            population.append(kromosomi)
            
        return population
    
    def tulostaKromosomi(self, kromosomi, score = 0):
        
        C = 0.0
        gamma = 0.0                    
        muut = ""
        
        paikka = 0
        if kromosomi[paikka] == 0:
            muut = 'ovr'
        else:
            muut = 'ovo'
        paikka = paikka + MUUT           
            
        gamma = binFloatiksi(kromosomi, paikka,paikka+GAMMA_INT, paikka+GAMMA_INT+GAMMA_FLOAT)
        paikka = paikka + GAMMA_INT + GAMMA_FLOAT
        
        C = binFloatiksi(kromosomi,paikka ,paikka + C_INT ,paikka + C_INT+C_FLOAT)
        paikka = paikka + C_INT + C_FLOAT
        
                    
        print("gamma: {:12.10f}   C: {:16.10f}  {}  tulos: {:7.7%}   ".format(gamma, C, muut, score))  
    
    
    def luoUusi(self):
        
        kromosomi = np.zeros(self.n_genes, dtype=np.int)            
        mask = np.random.rand(len(kromosomi)) < GEENIALUSTUS
        kromosomi[mask] = 1
        
        return kromosomi
    
    
    def luoParametrit(self, kromosomi):
        
        muut = []
        gamma = []
        C = []
        painot = []
        
        paikka = 0
        if kromosomi[paikka] == 0:
            muut.append('ovr')
        else:
            muut.append('ovo')
        paikka = paikka + MUUT           
            
        gamma.append(binFloatiksi(kromosomi, paikka,paikka+GAMMA_INT, paikka+GAMMA_INT+GAMMA_FLOAT))                
        paikka = paikka + GAMMA_INT + GAMMA_FLOAT
        
        C.append(binFloatiksi(kromosomi,paikka ,paikka + C_INT ,paikka + C_INT+C_FLOAT))
        paikka = paikka + C_INT + C_FLOAT
        
        if self.luokkapainot:
            painot.append('balanced')
        else:
            painot.append(None)
                    
        paradict = {'C' : C, 'gamma' : gamma, 'decision_function_shape' : muut, 'class_weight' : painot}
        
        return paradict
    
    
    def fitness(self, population):
        X, y = self.dataset
        scores = []                
        paraLista = []
            
        population = np.unique(population, axis = 0)
        
        for i in range(self.size - len(population)):
            population = np.vstack((population, self.luoUusi()))
        
        for kromosomi in population:            
            paraLista.append(self.luoParametrit(kromosomi))
                        
        haku = GridSearchCV(SVC(), refit = False, param_grid = paraLista, cv = self.cv, verbose = 0, n_jobs = -2, iid = self.cvid)
        haku.fit(X, y)
        
        if self.verbose >2:
            for mean, params in zip(haku.cv_results_['mean_test_score'],  haku.cv_results_['params']):                
                print("Parametrit: {}            Tulos: {}".format(params, mean))
                            
        scores = haku.cv_results_['mean_test_score']                                                                                   
        scores, population = np.array(scores), np.array(population)
        inds = np.flip(np.argsort(scores), axis = 0)                        
        return list(scores[inds]), list(population[inds])
    

    def select(self, population_sorted):
        
        population_next = []
        
        for i in range(0, self.n_best):            
            population_next.append(population_sorted[i])
            
        for i in range(0, self.n_rand):
            population_next.append(random.choice(population_sorted))
                    
        return population_next


    def crossover(self, population, ):
        
        population_next = []
        
        for i in range(0, len(population)):
            
            kumppanit = list(range(len(population)))
            kumppanit.remove(i)
            
            if len(self.eliittikerroin) > 0 and i < len(self.eliittikerroin):                
                kumppanit = random.sample(kumppanit, self.eliittikerroin[i])
                
            else:
                kumppanit = random.sample(kumppanit, self.n_children)
            
            for kumppani in kumppanit:
                
                isa, aiti = np.array(population[i]), np.array(population[kumppani])
                lapsi = isa
                maski = np.random.rand(len(lapsi)) < GEENISIIRTO
                lapsi[maski] = aiti[maski]                
                population_next.append(lapsi)
                
        return population_next
                
    
    def mutate(self, population, kaikki = False):
        
        population_next = []
        
        if kaikki == False:
            
            for i in range(len(population)):
                
                kromosomi = np.array(population[i])
                
                if random.random() < self.mutation_rate:
                                               
                     randomit = np.array(np.random.randint(2, size = len(kromosomi)))
                     mask = np.random.rand(len(kromosomi)) < YHDENMUTRATE
                     kromosomi[mask] = randomit[mask]  
                   
                population_next.append(kromosomi)
        else:
            
            if self.verbose>1:                
                print("mUtanTTi-invAAsiO!")            
                
            for i in range(len(population)):
                
                kromosomi = np.array(population[i])                                      
                randomit = np.array(np.random.randint(2, size = len(kromosomi)))
                mask = np.random.rand(len(kromosomi)) < YHDENMUTRATE
                kromosomi[mask] = randomit[mask]                    
                population_next.append(kromosomi)
                        
        return population_next



    def generate(self, population):
        
        geneaika = time.time()
        
        # valinta, lisääntyminen, mutaatio
        scores_sorted, population_sorted = self.fitness(population)        
        population = self.select(population_sorted)
        population = self.crossover(population)
        population = self.mutate(population)
        
        # Historiaa
        self.kromosomis_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))
        self.generation = self.generation + 1
                
        if self.verbose > 0:
            print("")
            print("Sukupolvi {:4d}".format(self.generation))   
                
        if self.verbose == 2:            
            print("Parhaat jatkoon")
            
            for i in range(self.n_best+1):
                
                score = scores_sorted[i]
                self.tulostaKromosomi(population_sorted[i], score)
                
        geneloppuaika = time.time()                                                
        
        if self.verbose > 0:
            print("Tulos: {:7.5%}   Keskiarvo: {:7.5%}   Keskihajonta: {:7.5%}".format(scores_sorted[0],np.mean(scores_sorted), np.std(scores_sorted)))             
            print("Aikaa meni sukupolvelle: {:5.3f} min".format((geneloppuaika - geneaika)/60)) 
                         
                    
        # Mutatatoidaan koko populaatio, jos alle std rajan heitto                
        if np.std(scores_sorted) < INVAASIORAJA and self.generation > 3:            
            population = self.mutate(population, True)
                             
        if scores_sorted[0] > self.paras:
                        
            self.paras = scores_sorted[0]
            self.parasKromosomi = population_sorted[0]
            
            if self.verbose >0:              
                print("")
                print("*** UUSI PARAS ***")            
                self.tulostaKromosomi(self.parasKromosomi, self.paras)               
                print("")
                
        return population

    
    def tuoEstimaattori(self):
        
        parasDict = self.luoParametrit(self.parasKromosomi)
        
        C = parasDict['C']
        gamma = parasDict['gamma']        
        muut = parasDict['decision_function_shape']                
        C = C[0]
        gamma = gamma[0]
        muut = muut[0]
        
        if self.luokkapainot:
            painot = 'balanced'
        else:
            painot = None
        
        
        estimaattori = SVC(C = C, gamma = gamma, decision_function_shape = muut, class_weight = painot)

        return estimaattori


    def fit(self, X, y, kierroksia):
        
        start = time.time()
        aikaaYhteensa =  time.time()
        for j in range(kierroksia):
            start = time.time()
                        
            self.maailmanParas = 0

                  
            
            if self.verbose > 0:
                print("######### UUSI MAAILMA nro:{:3d} #########".format(j+1))
            
            
            self.scores_best, self.scores_avg  = [], []
            
            self.dataset = X, y
            
            self.n_genes = MUUT + GAMMA_INT + GAMMA_FLOAT + C_INT + C_FLOAT
            
            population = self.initilize()
            
            for i in range(self.n_gen):
                
                population = self.generate(population)
                  
            self.generation = 0                                          
            self.parhaat.append(self.paras)
                        
            
            if self.tulostetaanAikakausi:
                print("Paras tulos tähän asti:")
                self.tulostaKromosomi(self.parasKromosomi, self.paras)
                self.plot_scores()
             
            end = time.time()
            print("Aikaa meni: {:5.3f} min".format((end - start)/60)) 
            print("Paras löydetty kromosomi:")
            self.tulostaKromosomi(self.parasKromosomi, self.paras)  
            
        aikaaLopussa = time.time()
        print("Aikaa meni yhteensä: {:5.3f} min".format((aikaaLopussa - aikaaYhteensa)/60))             
            
        
        if self.tulostetaanKehitys:
            self.plot_parhaat()
                 
    @property
    def support_(self):
        return self.kromosomis_best[-1]

      
    def plot_parhaat(self):        
        plt.figure(figsize=(30,20))        
        plt.plot(self.parhaat, label='Paras')        
        plt.legend()
        plt.title("Parhaan tuloksen kehitys maailmoiden suhteen")
        plt.ylabel('Paras tulos')
        plt.xlabel('Maailma nro.')        
        plt.show()

    def plot_scores(self):        
        plt.figure(figsize=(30,20))        
        plt.plot(self.scores_best, label='Paras')
        plt.plot(self.scores_avg, label='Keskiarvo')
        plt.legend()
        plt.ylabel('Tulos')
        plt.xlabel('Sukupolvi')        
        plt.show()
