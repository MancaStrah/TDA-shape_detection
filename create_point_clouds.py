import numpy as np
import pickle

#DEFINING DIFFERENT SHAPES:
#TO DO:verjetno je treba malo spremeniti določene oblike, točke je treba še malo pretresti, trenutno so res narejene da točno
#sledijo obliki, tega nočemo, morajo malo odstopati v vseh treh dimenzijah
#fino bi bilo zgenerirati z različnim številom točk (ne premajhnim, ker bo to pokvarilo model, stvar mora še vedno izgledati kot krogla/črta...),
#preveč točk ne sme biti, ker potem predolgo računa stvari
#spremeniti te radije, da imamo različno velike
#tu je mišljeno da se vse te oblike zgenerira enkrat, hkrati in se jih shrani v spodaj napisano datoteko. 
#Tako da bo treba morda popraviti kak del kode, tako da bo generiralo npr različno velike krogle in črte, ne da bi morali na roko 
#spreminjati ta parameter (torej treba bo neke dodati neke spremenljivke ki bodo določale različne velikosti ipd.)

def create_sphere(num_points=300, radius=1.0, rand=0.1):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.sin(phi) * np.cos(theta) + randx
    y = radius * np.sin(phi) * np.sin(theta) + randy
    z = radius * np.cos(phi) + randz
    return np.column_stack((x, y, z))

def create_circle(num_points=300, radius=1.0, rand=0.1):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.cos(theta) + randx
    y = radius * np.sin(theta) + randy
    z = np.zeros(num_points) + randz
    return np.column_stack((x, y, z))

def create_line_segment(num_points=100, length=1.0, rand=0.1):
    x = np.random.uniform(0, length, num_points) 
    y = np.random.uniform(-rand, rand, num_points)
    z = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_torus(num_points=300, R=1.0, r=0.3, rand=0.1):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(phi)) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    y = (R + r * np.cos(phi)) * np.sin(theta) + np.random.uniform(-rand, rand, num_points)
    z = r * np.sin(phi) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_flat_disc(num_points=300, radius=1.0, rand=0.1):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.random.rand(num_points) * np.cos(theta)
    y = radius * np.random.rand(num_points) * np.sin(theta)
    z = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_ellipsoid(num_points=300, a=1.0, b=0.8, c=0.6):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi) 
    return np.column_stack((x, y, z))

def create_perturbed_3_disc(num_points=300, perturbation=0.5):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    x_disc = np.sin(theta) * np.cos(phi)
    y_disc = np.sin(theta) * np.sin(phi)
    z_disc = np.cos(theta)
    # Deform the 3-disc to create a perturbed shape
    x_perturbed = x_disc + perturbation * np.random.normal(size=num_points)
    y_perturbed = y_disc + perturbation * np.random.normal(size=num_points)
    z_perturbed = z_disc + perturbation * np.random.normal(size=num_points)
    return np.column_stack((x_perturbed, y_perturbed, z_perturbed))



#tu notri dodaš zadeve, ki jih na novo napišeš, vse te oblike bo generiralo in jih tudi označilo z 0,1,2...glede na obliko
#kako je označena kakšna oblika si lahko pogledaš v label_mapping.txt
def main():
    shape_generators = {
        'sphere': create_sphere,
        'circle': create_circle,
        'line_segment': create_line_segment,
        'torus': create_torus,
        'flat_disc': create_flat_disc,
        'ellipsoid': create_ellipsoid,
        'perturbed_3_disc': create_perturbed_3_disc
    }

    shape_data = []
    label_mapping = {}

    for label, generator in enumerate(shape_generators.items()):
        shape_name, shape_func = generator
        label_mapping[shape_name] = label
        #tu je koliko vsake oblike bo zgeneriralo (to da je vsake oblike enako veliko je dobro za trening modela)
        for _ in range(5):
            point_cloud = shape_func()
            shape_data.append((point_cloud, label))

    with open('Shape_detection/shapes_data.pkl', 'wb') as file:
        pickle.dump(shape_data, file)

    with open('Shape_detection/label_mapping.txt', 'w') as file:
        for shape_name, label in label_mapping.items():
            file.write(f"{shape_name}:{label}\n")

    print("Shapes and labels saved successfully.")

if __name__ == "__main__":
    main()
