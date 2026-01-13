#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# POSTAVKE ALGORITMA
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.8
ELITISM_SIZE = 5

# PARAMETRI ZA PUNJAČE
MIN_CHARGERS_PER_CITY = 1  # Nijedan grad bez punjača
MAX_CHARGERS_PER_CITY = 15
MIN_SPACING = 15  # Minimalna distanca između punjača u km

@dataclass
class City:
    """Predstavlja grad u Crnoj Gori"""
    name: str
    population: int
    latitude: float  # Geografska širina (N)
    longitude: float  # Geografska dužina (E)
    vehicles: int = 0  # procenjeni broj vozila
    traffic_intensity: float = 0  # intenzitet saobraćaja
    
    def __post_init__(self):
        # Procenjeni broj vozila = 15% populacije
        self.vehicles = int(self.population * 0.15)
        # Intenzitet saobraćaja proporcionalan populaciji
        self.traffic_intensity = self.population / 1000

class MontenegroEVCS:
    """Glavna klasa za optimizaciju rasporeda punjača"""
    
    def __init__(self):
        """Inicijalizuje podatke o Crnoj Gori sa tačnim koordinatama"""
        self.cities = self._create_cities()
        self.candidate_locations = self._create_candidate_locations()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        
    def _create_cities(self) -> List[City]:
        """Kreira listu opština sa popisom 2023 (623 633 stanovnika) i GPS koordinatama."""
        # Popis 2023 (slika koju si poslao)
        cities_data = {
            'Andrijevica': (3910, 42.7333, 19.7919),
            'Bar':         (45812, 42.0937, 19.0984),
            'Berane':      (24645, 42.8425, 19.8733),
            'Bijelo Polje':(38662, 43.0383, 19.7476),
            'Budva':       (27445, 42.2864, 18.8400),
            'Cetinje':     (14494, 42.3906, 18.9142),
            'Danilovgrad': (18617, 42.5538, 19.1461),
            'Gusinje':     (3933, 42.5619, 19.8369),
            'Herceg Novi': (30824, 42.4531, 18.5375),
            'Kolašin':     (6700, 42.8225, 19.5167),
            'Kotor':       (22746, 42.4207, 18.7683),
            'Mojkovac':    (6728, 42.9604, 19.5833),
            'Nikšić':      (65705, 42.7731, 18.9445),
            'Petnjica':    (4957, 42.8822, 19.9975),
            'Plav':        (9050, 42.5969, 19.9458),
            'Pljevlja':    (24134, 43.3567, 19.3584),
            'Plužine':     (2137, 43.1531, 18.8392),
            'Podgorica':   (179505, 42.4411, 19.2636),
            'Rožaje':      (23184, 42.8330, 20.1665),
            'Šavnik':      (1569, 42.9569, 19.0964),
            'Tivat':       (16338, 42.4362, 18.6936),
            'Tuzi':        (12979, 42.3650, 19.3314),
            'Ulcinj':      (20507, 41.9294, 19.2244),
            'Žabljak':     (2941, 43.1564, 19.0775),
            'Zeta':        (16071, 42.3080, 19.2310),
        }

        cities: List[City] = []
        for name, (pop, lat, lon) in cities_data.items():
            cities.append(City(name=name, population=pop, latitude=lat, longitude=lon))

        return sorted(cities, key=lambda c: c.population, reverse=True)

    
    def _create_candidate_locations(self) -> List[Tuple[float, float]]:
        """Kreira kandidatske lokacije između gradova na osnovu stvarne geografije"""
        locations = []
        
        # Dodaj sve gradove kao kandidatske lokacije
        for city in self.cities:
            locations.append((city.latitude, city.longitude))
        
        # Dodaj dodatne lokacije između gradova (duž putnih ruta)
        for i, city1 in enumerate(self.cities):
            for city2 in self.cities[i+1:]:
                # Kreiraj 2-3 dodatne lokacije između svaka dva grada
                for k in range(1, 3):
                    t = k / 3.0
                    lat = city1.latitude + t * (city2.latitude - city1.latitude)
                    lon = city1.longitude + t * (city2.longitude - city1.longitude)
                    # Dodaj malu nasumičnu devijaciju
                    lat += np.random.uniform(-0.1, 0.1)
                    lon += np.random.uniform(-0.1, 0.1)
                    locations.append((lat, lon))
        
        return list(set(locations))  # Ukloni duplikate
    
    def _calculate_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """
        Izračunava rastojanje između dve GPS tačke koristeći Haversine formulu
        Rezultat je u kilometrima
        """
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        # Konvertuj u radijane
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radijus Zemlje u km
        earth_radius = 6371
        
        # Osnovana distanca
        distance = earth_radius * c
        
        # Korekcija za planinsku geografiju Crne Gore
        road_factor = 1.15
        
        return distance * road_factor
    
    def _calculate_fitness(self, chargers: List[Tuple[int, int]]) -> float:
        """
        Izračunava fitness funkciju za dati raspored punjača
        
        Komponente:
        1. Pokrivenost populacije (Consumer Satisfaction)
        2. Minimalna distanca između punjača
        3. Ravnomerna distribucija
        4. Pokrivenost saobraćaja
        5. Ograničenje
        """
        if not chargers:
            return float('-inf')
        
        charger_locations = [self.candidate_locations[idx] for idx, _ in chargers]
        
        fitness = 0
        
        # 1. POKRIVENOST POPULACIJE
        population_satisfaction = 0
        for city in self.cities:
            # Pronađi najbliži punjač
            min_dist = float('inf')
            for charger_loc in charger_locations:
                dist = self._calculate_distance((city.latitude, city.longitude), charger_loc)
                min_dist = min(min_dist, dist)
            
            # Nagrada je inverzna distanci
            satisfaction = 1.0 / (1.0 + min_dist / 50)
            # Veća nagrada za velike gradove
            satisfaction *= (1.0 + np.log1p(city.population / 10000))
            population_satisfaction += satisfaction * (city.population / 1000)
        
        fitness += population_satisfaction * 0.35
        
        # 2. MINIMALNA DISTANCA IZMEĐU PUNJAČA (sprečava dupliranje)
        spacing_penalty = 0
        for i, loc1 in enumerate(charger_locations):
            for loc2 in charger_locations[i+1:]:
                dist = self._calculate_distance(loc1, loc2)
                if dist < MIN_SPACING:
                    spacing_penalty += (MIN_SPACING - dist) * 2
        
        fitness -= spacing_penalty * 0.25
        
        # 3. RAVNOMERNA DISTRIBUCIJA (ne smiju biti sve na istom mjestu)
        if len(chargers) > 1:
            # Izračunaj varijansu lokacija
            lat_coords = [loc[0] for loc in charger_locations]
            lon_coords = [loc[1] for loc in charger_locations]
            lat_variance = np.var(lat_coords) if len(lat_coords) > 1 else 0
            lon_variance = np.var(lon_coords) if len(lon_coords) > 1 else 0
            distribution_score = (lat_variance + lon_variance) / 2
            fitness += min(distribution_score, 1.0) * 0.15
        
        # 4. POKRIVENOST SAOBRAĆAJA (prioritet rutama sa više vozila)
        traffic_score = 0
        for city in self.cities:
            min_dist = float('inf')
            for charger_loc in charger_locations:
                dist = self._calculate_distance((city.latitude, city.longitude), charger_loc)
                min_dist = min(min_dist, dist)
            
            # Nagrada ako je punjač blizu gradova sa intenzivnim saobraćajem
            if min_dist < 30:
                traffic_score += city.traffic_intensity
        
        fitness += traffic_score * 0.15
        
        # 5. OGRANIČENJE: Svaki grad mora imati bar jedan punjač na razumnoj distanci
        coverage_constraint = 0
        for city in self.cities:
            min_dist = min([self._calculate_distance((city.latitude, city.longitude), loc) 
                           for loc in charger_locations])
            if min_dist > 50:  # Kazna ako se grad nalazi dalje od 50 km
                coverage_constraint -= 10
        
        fitness += coverage_constraint
        
        return fitness
    
    def _create_solution(self) -> List[Tuple[int, int]]:
        """Kreira nasumičnu soluciju - listu (indeks_lokacije, kapacitet)"""
        
        # Broj punjača = 1 za svaki 20,000 stanovnika
        total_population = sum(c.population for c in self.cities)
        num_chargers = int(total_population / 20000)
        
        solution = []
        used_locations = set()
        
        # PRIORITET 1: Podgorica - NAJVEĆI GRAD - 6-8 stanica
        podgorica = self.cities[0]  # Podgorica je prvi (sortiran po populaciji)
        num_podgorica = 6  # KRUTO - minimum 6 stanica u Podgorici!
        
        for _ in range(num_podgorica):
            # Pronađi najbližu NEISKORIŠĆENU lokaciju
            for _ in range(5):  # Pokušaj 5 puta da nađeš drugu
                closest_idx = min(
                    (i for i in range(len(self.candidate_locations)) if i not in used_locations),
                    key=lambda i: self._calculate_distance(
                        (podgorica.latitude, podgorica.longitude),
                        self.candidate_locations[i]),
                    default=None
                )
                if closest_idx is not None and closest_idx not in used_locations:
                    capacity = random.randint(4, 6)  # Veći kapacitet za Podgoricu
                    solution.append((closest_idx, capacity))
                    used_locations.add(closest_idx)
                    break
        
        # PRIORITET 2: Ostali veliki gradovi - 2-3 stanice po gradu
        for city in self.cities[1:10]:  # Gradovi 2-10 po veličini
            num_for_city = max(2, int(city.population / 40000))  # Min 2 stanice
            
            for _ in range(num_for_city):
                closest_idx = min(
                    (i for i in range(len(self.candidate_locations)) if i not in used_locations),
                    key=lambda i: self._calculate_distance(
                        (city.latitude, city.longitude),
                        self.candidate_locations[i]),
                    default=None
                )
                if closest_idx is not None and closest_idx not in used_locations:
                    capacity = random.randint(2, 4)
                    solution.append((closest_idx, capacity))
                    used_locations.add(closest_idx)
                    if len(solution) >= num_chargers:
                        break
            
            if len(solution) >= num_chargers:
                break
        
        # PRIORITET 3: Mali gradovi - 1 stanica
        for city in self.cities[10:]:
            if len(solution) >= num_chargers:
                break
            
            closest_idx = min(
                (i for i in range(len(self.candidate_locations)) if i not in used_locations),
                key=lambda i: self._calculate_distance(
                    (city.latitude, city.longitude),
                    self.candidate_locations[i]),
                default=None
            )
            if closest_idx is not None and closest_idx not in used_locations:
                capacity = random.randint(1, 2)
                solution.append((closest_idx, capacity))
                used_locations.add(closest_idx)
        
        # PRIORITET 4: Popuni ostatak na magistralnim rutama
        while len(solution) < num_chargers:
            idx = random.randint(0, len(self.candidate_locations)-1)
            if idx not in used_locations:
                capacity = random.randint(1, 3)
                solution.append((idx, capacity))
                used_locations.add(idx)
        
        # Ukloni duplikate (trebalo bi da ih nema sada)
        seen = set()
        unique_solution = []
        for item in solution:
            if item[0] not in seen:
                seen.add(item[0])
                unique_solution.append(item)
        
        return unique_solution[:num_chargers]

    
    def _mutate(self, solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Mutira soluciju"""
        mutated = solution.copy()
        
        if random.random() < MUTATION_RATE and len(mutated) > 0:
            # Mutacija 1: Ukloni punjač sa najmanjim kapacitetom
            if random.random() < 0.5:
                min_idx = min(range(len(mutated)), 
                            key=lambda i: mutated[i][1])
                mutated.pop(min_idx)
        
        if random.random() < MUTATION_RATE:
            # Mutacija 2: Dodaj novi punjač
            idx = random.randint(0, len(self.candidate_locations)-1)
            capacity = random.randint(1, 6)
            mutated.append((idx, capacity))
        
        if random.random() < MUTATION_RATE and len(mutated) > 0:
            # Mutacija 3: Promeni kapacitet postojećeg punjača
            random_idx = random.randint(0, len(mutated)-1)
            loc_idx, _ = mutated[random_idx]
            new_capacity = random.randint(1, 6)
            mutated[random_idx] = (loc_idx, new_capacity)
        
        # Ukloni duplikate
        seen = set()
        unique_solution = []
        for item in mutated:
            if item[0] not in seen:
                seen.add(item[0])
                unique_solution.append(item)
        
        return unique_solution
    
    def _crossover(self, parent1: List[Tuple[int, int]], 
                  parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Crossing over između dva roditelja (REX - Regional Exchange Crossover)"""
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1.copy() if parent1 else parent2.copy()
        
        # Podeli gradove na tri regije (sever, sredina, jug) po geografskoj širini
        lat_coords = [self.candidate_locations[idx][0] for idx, _ in parent1] + \
                     [self.candidate_locations[idx][0] for idx, _ in parent2]
        lat_min, lat_max = min(lat_coords), max(lat_coords)
        lat_third = (lat_max - lat_min) / 3
        
        child = []
        
        # Za svaku lokaciju u parent1
        for idx, cap in parent1:
            lat = self.candidate_locations[idx][0]
            
            if lat < lat_min + lat_third:
                # Severna regija - uzmi iz parent1
                child.append((idx, cap))
            elif lat < lat_min + 2*lat_third:
                # Srednja regija - uzmi iz parent2
                if any(i == idx for i, _ in parent2):
                    cap2 = next(c for i, c in parent2 if i == idx)
                    child.append((idx, cap2))
            # Južna regija - opet parent1
        
        # Dodaj lokacije iz parent2 koje nedostaju
        for idx, cap in parent2:
            if not any(i == idx for i, _ in child):
                lat = self.candidate_locations[idx][0]
                if lat >= lat_min + 2*lat_third:  # Južna regija
                    child.append((idx, cap))
        
        return child if child else parent1.copy()
    
    def optimize(self):
        """Pokreće genetski algoritam"""
        print("\n" + "="*80)
        print("OPTIMIZACIJA RASPOREDA ELEKTRČNIH PUNJAČA - CRNA GORA")
        print("="*80)
        
        # Kreiraj početnu populaciju
        population = [self._create_solution() for _ in range(POPULATION_SIZE)]
        
        print(f"\nParametri algoritma:")
        print(f"  - Veličina populacije: {POPULATION_SIZE}")
        print(f"  - Broj generacija: {GENERATIONS}")
        print(f"  - Stopa mutacije: {MUTATION_RATE}")
        print(f"  - Broj kandidatskih lokacija: {len(self.candidate_locations)}")
        print(f"  - Broj gradova: {len(self.cities)}")
        print(f"\nGradovi (sa TAČNIM geografskim koordinatama):")
        for i, city in enumerate(self.cities[:5], 1):
            print(f"  {i}. {city.name:15s}: {city.population:>7,} st. | "
                  f"Lat: {city.latitude:7.4f}, Lon: {city.longitude:7.4f}")
        print(f"  ... i još {len(self.cities)-5} gradova\n")
        
        # Genetski algoritam - glavna petlja
        for gen in range(GENERATIONS):
            # Izračunaj fitness za sve u populaciji
            fitness_scores = [(self._calculate_fitness(sol), sol) for sol in population]
            fitness_scores.sort(reverse=True)
            
            # Spremi良iju soluciju
            best_fitness, best_sol = fitness_scores[0]
            self.fitness_history.append(best_fitness)
            
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_solution = best_sol
            
            # Ispis napretka
            if (gen + 1) % 10 == 0 or gen == 0:
                avg_fitness = np.mean([f for f, _ in fitness_scores])
                print(f"Generacija {gen+1:3d}/{GENERATIONS} | "
                      f"Best: {best_fitness:8.2f} | "
                      f"Avg: {avg_fitness:8.2f} | "
                      f"Broj punjača: {len(best_sol)}")
            
            # Kreiraj novu populaciju
            new_population = []
            
            # Elitism - predrži najbolje rešenje
            for _ in range(ELITISM_SIZE):
                new_population.append(fitness_scores[_][1])
            
            # Genetske operacije
            while len(new_population) < POPULATION_SIZE:
                # Turnirska selekcija
                candidates_idx = random.sample(range(len(fitness_scores)), min(5, len(fitness_scores)))
                parent1 = fitness_scores[min(candidates_idx)][1]
                
                candidates_idx = random.sample(range(len(fitness_scores)), min(5, len(fitness_scores)))
                parent2 = fitness_scores[min(candidates_idx)][1]
                
                # Crossover
                if random.random() < CROSSOVER_RATE:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutacija
                if random.random() < MUTATION_RATE:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population[:POPULATION_SIZE]
        
        print("\n" + "="*80)
        print(f"OPTIMIZACIJA ZAVRŠENA!")
        print(f"Najbolji fitness: {self.best_fitness:.2f}")
        print(f"Preporučeni broj punjača: {len(self.best_solution)}")
        print("="*80 + "\n")
        
        return self.best_solution
    
    def visualize(self):
        """Vizuelizuje rezultate na mapi Crne Gore sa TAČNIM koordinatama"""
        # Kreiraj figure sa side-by-side layout (mapa + legenda + info)
        fig = plt.figure(figsize=(14, 10), dpi=150)
        
        # Mapa će biti lijeva strana
        ax_map = plt.subplot(121)
        
        # Postavi granice (precizno za Crnu Goru)
        # Crna Gora: lat 41.8 - 43.6, lon 18.4 - 20.4
        ax_map.set_xlim(18.3, 20.5)  # Geografska dužina (Longitude)
        ax_map.set_ylim(41.8, 43.7)  # Geografska širina (Latitude)
        ax_map.set_aspect('equal')
        
        ax_map.set_title(f'OPTIMALNI RASPORED ELEKTRČNIH PUNJAČA\nCRNA GORA',
                     fontsize=14, fontweight='bold', pad=12)
        ax_map.set_xlabel('Geografska dužina (°E)', fontsize=10, fontweight='bold')
        ax_map.set_ylabel('Geografska širina (°N)', fontsize=10, fontweight='bold')
        
        # Crtanje mreže
        ax_map.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
        
        # 1. CRTAJ POTENCIJALNE LOKACIJE (male plave zvezdice)
        for loc_idx, loc in enumerate(self.candidate_locations):
            ax_map.scatter(loc[1], loc[0], marker='*', s=25, 
                      color='lightblue', alpha=0.25, edgecolors='none', zorder=1)
        
        # 2. CRTAJ GRADOVE (krugovi proporcionalni populaciji)
        max_pop = max(c.population for c in self.cities)
        min_pop = min(c.population for c in self.cities)
        
        for city in self.cities:
            # Veličina kruga proporcionalna populaciji
            size = 80 + (city.population - min_pop) / (max_pop - min_pop) * 700
            
            # Boja proporcionalna broju vozila
            color_intensity = city.vehicles / max(c.vehicles for c in self.cities)
            
            ax_map.scatter(city.longitude, city.latitude, s=size, alpha=0.65, 
                      c=[color_intensity], cmap='YlOrRd',
                      edgecolors='black', linewidth=1.5, zorder=3)
            
            # Dodaj naziv grada
            ax_map.annotate(city.name, (city.longitude, city.latitude), 
                       xytext=(4, 4), textcoords='offset points',
                       fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.55))
        
        # 3. CRTAJ ODABRANE LOKACIJE PUNJAČA (crveni X)
        if self.best_solution:
            for loc_idx, capacity in self.best_solution:
                loc = self.candidate_locations[loc_idx]
                
                # Crveni X
                ax_map.plot(loc[1], loc[0], marker='x', markersize=14, 
                       color='red', markeredgewidth=3, linestyle='none',
                       zorder=5)
                
                # Dodaj kapacitet kod X
                ax_map.annotate(f'{capacity}', (loc[1], loc[0]),
                           xytext=(7, -9), textcoords='offset points',
                           fontsize=6, color='darkred', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.15', 
                                    facecolor='white', edgecolor='red', linewidth=1))
        
        # ===== DESNA STRANA: LEGENDA I STATISTIKA =====
        ax_info = plt.subplot(122)
        ax_info.axis('off')
        
        # Naslov
        title_text = f"Preporučeni broj punjača: {len(self.best_solution)}\nKvaliteta rešenja: {self.best_fitness:.2f}"
        ax_info.text(0.5, 0.95, title_text, transform=ax_info.transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='black', linewidth=1.5))
        
        # LEGENDA
        legend_y = 0.85
        ax_info.text(0.05, legend_y, 'LEGENDA:', transform=ax_info.transAxes,
                    fontsize=11, fontweight='bold', va='top')
        
        # Element 1: Gradovi
        ax_info.add_patch(mpatches.Circle((0.08, legend_y-0.08), 0.015, 
                                         transform=ax_info.transAxes,
                                         facecolor='yellow', edgecolor='black', linewidth=1))
        ax_info.text(0.12, legend_y-0.078, 'Gradovi (veličina = populacija)', 
                    transform=ax_info.transAxes, fontsize=9, va='center')
        
        # Element 2: Punjači
        ax_info.plot([0.08], [legend_y-0.16], marker='x', markersize=12, 
                    color='red', markeredgewidth=2.5, linestyle='none',
                    transform=ax_info.transAxes)
        ax_info.text(0.12, legend_y-0.16, 'Lokacije punjača (X)', 
                    transform=ax_info.transAxes, fontsize=9, va='center')
        
        # Element 3: Potencijalne
        ax_info.scatter([0.08], [legend_y-0.24], marker='*', s=60, 
                       color='lightblue', alpha=0.5, edgecolors='none',
                       transform=ax_info.transAxes)
        ax_info.text(0.12, legend_y-0.24, 'Potencijalne lokacije (*)', 
                    transform=ax_info.transAxes, fontsize=9, va='center')
        
        # ===== STATISTIKA =====
        stats_y = legend_y - 0.35
        ax_info.text(0.05, stats_y, 'STATISTIKA:', transform=ax_info.transAxes,
                    fontsize=11, fontweight='bold', va='top')
        
        stats_text = f"""
Ukupna populacija: {sum(c.population for c in self.cities):,}
Ukupno vozila: {sum(c.vehicles for c in self.cities):,}

Min. kapacitet: {min([cap for _, cap in self.best_solution])} pile-ova
Max. kapacitet: {max([cap for _, cap in self.best_solution])} pile-ova
Prosečan: {np.mean([cap for _, cap in self.best_solution]):.1f} pile-ova

Broj gradova: {len(self.cities)}
Kandidatskih lokacija: {len(self.candidate_locations)}
"""
        
        ax_info.text(0.05, stats_y - 0.08, stats_text, transform=ax_info.transAxes,
                    fontsize=9, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6, edgecolor='orange'))
        
        plt.tight_layout()
        plt.savefig('montenegro_evcs_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Mapa je sačuvana kao 'montenegro_evcs_distribution.png'")
        plt.show()
    
    def print_detailed_report(self):
        """Štampa detaljni izveštaj"""
        print("\n" + "="*110)
        print("DETALJNI IZVEŠTAJ O OPTIMALNOM RASPOREDU PUNJAČA")
        print("="*110)
        
        charger_locations = [self.candidate_locations[idx] for idx, _ in self.best_solution]
        
        print(f"\nOPŠTE INFORMACIJE:")
        print(f"  - Broj punjača: {len(self.best_solution)}")
        print(f"  - Kvaliteta rešenja: {self.best_fitness:.2f}")
        print(f"  - Grad sa najvećom populacijom: {self.cities[0].name} ({self.cities[0].population:,} stanovnika)")
        print(f"  - Grad sa najmanjom populacijom: {self.cities[-1].name} ({self.cities[-1].population:,} stanovnika)")
        
        print(f"\nKAPACITETI PUNJAČA:")
        capacities = [cap for _, cap in self.best_solution]
        print(f"  - Minimalni kapacitet: {min(capacities)} pile-ova")
        print(f"  - Maksimalni kapacitet: {max(capacities)} pile-ova")
        print(f"  - Prosečan kapacitet: {np.mean(capacities):.1f} pile-ova")
        
        print(f"\nANALIZA POKRIVANJA PO GRADOVIMA:")
        print(f"{'Grad':<20} {'Distanca (km)':<15} {'Status':<20}")
        print("-" * 110)
        
        for city in self.cities:
            min_dist = min([self._calculate_distance((city.latitude, city.longitude), loc) 
                           for loc in charger_locations], default=float('inf'))
            
            if min_dist < 20:
                status = "✓ Odličan"
            elif min_dist < 40:
                status = "✓ Dobar"
            elif min_dist < 60:
                status = "⚠ Zadovoljavajući"
            else:
                status = "✗ Loš"
            
            print(f"{city.name:<20} {min_dist:<15.1f} {status:<20}")
        
        print(f"\nDISTANCE IZMEĐU PUNJAČA:")
        charger_coords = self.best_solution
        min_distances = []
        for i, (idx1, _) in enumerate(charger_coords):
            for idx2, _ in charger_coords[i+1:]:
                dist = self._calculate_distance(
                    self.candidate_locations[idx1],
                    self.candidate_locations[idx2]
                )
                min_distances.append(dist)
        
        if min_distances:
            print(f"  - Minimalna distanca: {min(min_distances):.1f} km")
            print(f"  - Prosečna distanca: {np.mean(min_distances):.1f} km")
            print(f"  - Maksimalna distanca: {max(min_distances):.1f} km")
        
        print("\n" + "="*110)

# GLAVNA FUNKCIJA
def main():
    """Pokreće optimizaciju"""
    
    # Kreiraj sistem
    system = MontenegroEVCS()
    
    # Pokreni optimizaciju
    best_solution = system.optimize()
    
    # Štampi detaljni izveštaj
    system.print_detailed_report()
    
    # Vizuelizuj rezultate
    system.visualize()
    
    # Spremi rezultate u fajl
    save_results(system)

def save_results(system):
    """Spremi rezultate u tekstualni fajl"""
    with open('REZULTATI_PUNJACA_CRNA_GORA.txt', 'w', encoding='utf-8') as f:
        f.write("="*110 + "\n")
        f.write("OPTIMALNI RASPORED ELEKTRČNIH PUNJAČA ZA CRNU GORU\n")
        f.write("="*110 + "\n\n")
        
        f.write(f"Datum generisanja: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
        f.write(f"Kvaliteta rešenja: {system.best_fitness:.2f}\n")
        f.write(f"Broj preporučenih punjača: {len(system.best_solution)}\n\n")
        
        f.write("PREPORUČENE LOKACIJE:\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'#':<3} {'Geografska širina':<20} {'Geografska dužina':<20} {'Kapacitet (pile)':<20}\n")
        f.write("-" * 110 + "\n")
        
        for i, (idx, capacity) in enumerate(system.best_solution, 1):
            lat, lon = system.candidate_locations[idx]
            f.write(f"{i:<3} {lat:<20.6f} {lon:<20.6f} {capacity:<20}\n")
        
        f.write("\n" + "="*110 + "\n")
        f.write("ANALIZA POKRIVANJA PO GRADOVIMA\n")
        f.write("="*110 + "\n")
        
        charger_locations = [system.candidate_locations[idx] for idx, _ in system.best_solution]
        for city in system.cities:
            min_dist = min([system._calculate_distance((city.latitude, city.longitude), loc) 
                           for loc in charger_locations], default=float('inf'))
            f.write(f"{city.name:20s}: {min_dist:6.1f} km do najbližeg punjača\n")
    
    print("✓ Rezultati su sačuvani u 'REZULTATI_PUNJACA_CRNA_GORA.txt'")

if __name__ == "__main__":
    main()