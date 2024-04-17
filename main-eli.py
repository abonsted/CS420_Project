import numpy as np
import hop

N = 32**2

def sign(x):
    if(x >= 0):
        return 1
    else:
        return -1

def imprint_single_image(image):
    weights = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            weights[i,j] += image[i] * image[j]

    weights *= 0.01

    for i in range(N):
        weights[i,i] = 0

    return weights

def async_recall(image, probe, weights):
    min_energy = 1000
    min_probe = []
    iter = 0

    while iter < 1000:
        neuron_index = np.random.randint(0, len(probe))
        probe[neuron_index] = sign(np.dot(weights[neuron_index], probe))

        energy = calculate_energy(probe, weights)

        if energy < min_energy:
            min_energy = energy
            min_probe = probe

            equal = np.array_equal(image, probe)

            if equal:
                return min_probe, min_energy, iter
            
        iter += 1

    return min_probe, min_energy, iter

def calculate_energy(probe, weights):
    energy = 0.0
    for i in range(N):
        for j in range(N):
            if(i != j):
                energy += weights[i,j]*probe[i]*probe[j]
    energy *= -0.5
    return energy

def noisy_probe(image, x):
    probe = image.copy()
    flip = np.arange(len(probe)) % x == 0
    probe[flip] *= -1
    return probe

def resemblance(image, probe):
    matching = np.sum(image == probe)
    total = image.shape[0]
    res = (matching / total) * 100
    return res


if __name__ == '__main__':
    image = hop.image_to_hopfield('test.jpg', 'BW')
    weights = imprint_single_image(image)
    probe = noisy_probe(image, 2)

    org_image = hop.hopfield_to_image(image, 'BW')
    org_image.save('org_image.jpg')

    noisy_image = hop.hopfield_to_image(probe, 'BW')
    noisy_image.save('noisy_image.jpg')

    print(f'Image vs. Noisy Probe: {resemblance(image, probe):.4f}%')

    new_probe, min_energy, iter = async_recall(image, probe, weights)

    print(f'Image vs. New Probe:   {resemblance(image, new_probe):.4f}%')

    new_image = hop.hopfield_to_image(new_probe, 'BW')
    new_image.save('new_image.jpg')