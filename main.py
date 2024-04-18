import numpy as np
import hop

N = 32**2

def imprint_images(images):
    weights = np.zeros((N, N))

    for image in images:
        image = image[:, np.newaxis]
        weights += np.dot(image, image.T)

    weights *= 0.01
    np.fill_diagonal(weights, 0)

    return weights

def async_recall(probe_, weights, max_iterations):
    probe = probe_.copy()
    energy_before = calculate_energy(probe, weights)

    for it in range(max_iterations):
        order = np.array(range(N))
        np.random.shuffle(order)

        for i in order:
            h = 0
            for j in range(weights.shape[0]):
                h += weights[i,j] * probe[j]

            probe[i] = -1 if h < 0 else 1

        energy_after = calculate_energy(probe, weights)

        if energy_after == energy_before:
            return probe, it
        
        energy_before = energy_after

    return probe, max_iterations

def sync_recall(probe, weights, max_iterations):
    energy_before = calculate_energy(probe, weights)

    for it in range(max_iterations):
        probe = np.dot(weights, probe)
        probe = np.where(probe >= 0, 1, -1)

        energy_after = calculate_energy(probe, weights)

        if energy_after == energy_before:
            return probe, it

        energy_before = energy_after

    return probe, max_iterations

def calculate_energy(probe, weights):
    return -0.5 * np.sum(weights * np.outer(probe, probe))

def noisy_probes(images, x):
    probes = []
    for image in images:
        probe = image.copy()
        flip = np.arange(len(probe)) % x == 0
        probe[flip] *= -1
        probes.append(probe)

    return np.array(probes)

def resemblance(image, probe):
    return (np.sum(image == probe) / image.shape[0]) * 100

def hamming_distance(original, recalled):
    return np.sum(original != recalled)

def test_images(num_images, recall='sync', save=False):
    images = hop.load_hopfield('images/test')
    images = images[:num_images]

    weights = imprint_images(images)
    probes = noisy_probes(images, 4)

    if save:
        for i in range(len(images)):
            orig = hop.hopfield_to_image(images[i])
            orig.save(f'images/orig/orig_{i}.jpg')

            noise = hop.hopfield_to_image(probes[i])
            noise.save(f'images/noise/noise_{i}.jpg')


    for i in range(len(images)):
        if recall == 'sync':
            new_probe, it = sync_recall(probes[i], weights, 1000)
        elif recall == 'async':
            new_probe, it = async_recall(probes[i], weights, 1000)

        if save:
            new = hop.hopfield_to_image(new_probe)
            new.save(f'images/new/new_{i}.jpg')

        noise_res = resemblance(images[i], probes[i])
        new_res = resemblance(images[i], new_probe)

        print(f'Image {i} ({it} iterations)')
        print(f'  Noise: {noise_res:.2f}%')
        print(f'  New:   {new_res:.2f}%')
        print()

def test_all():
    for num_images in range(1, 9):
        print(f'Testing {num_images} images')
        print()
        test_images(num_images)
        print()
        print()


if __name__ == '__main__':
    # test_all()

    test_images(8)