// Renders a lit box on a floor through the C API and checks the image is sane.
#include "bgrender.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond, msg)                                         \
    do                                                           \
    {                                                            \
        if (!(cond))                                             \
        {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                            \
        }                                                        \
    } while (0)

namespace
{

// Appends an axis-aligned quad as two triangles with a flat normal.
void quad(std::vector<float>& pos, std::vector<float>& nrm, std::vector<uint32_t>& idx, std::vector<uint32_t>& mat,
          const float c[4][3], const float n[3], uint32_t material)
{
    const uint32_t base = static_cast<uint32_t>(pos.size() / 3);
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k)
        {
            pos.push_back(c[i][k]);
            nrm.push_back(n[k]);
        }
    const uint32_t tris[6] = {0, 1, 2, 0, 2, 3};
    for (uint32_t t : tris)
        idx.push_back(base + t);
    mat.push_back(material);
    mat.push_back(material);
}

}  // namespace

int main()
{
    char error[512] = {0};
    BgrRenderer* r = bgr_create(BGR_MODULE_PATH, 0, error, sizeof(error));
    CHECK(r != nullptr, error);
    std::printf("device: %s\n", bgr_device_name(r));

    std::vector<float> pos, nrm;
    std::vector<uint32_t> idx, mat;
    const float floor_c[4][3] = {{-5, 0, -5}, {5, 0, -5}, {5, 0, 5}, {-5, 0, 5}};
    const float up[3] = {0, 1, 0};
    quad(pos, nrm, idx, mat, floor_c, up, 0);
    const float top_c[4][3] = {{-1, 1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}};
    quad(pos, nrm, idx, mat, top_c, up, 1);
    const float front_c[4][3] = {{-1, 0, 1}, {1, 0, 1}, {1, 1, 1}, {-1, 1, 1}};
    const float fwd[3] = {0, 0, 1};
    quad(pos, nrm, idx, mat, front_c, fwd, 1);

    BgrMaterial materials[2] = {
        {{0.2f, 0.5f, 0.2f}, 0.1f, 20.f, 0.0f, {0, 0, 0}, BGR_PATTERN_FELT},
        {{0.8f, 0.2f, 0.1f}, 0.6f, 60.f, 0.2f, {0, 0, 0}, BGR_PATTERN_NONE},
    };

    // Rendering before a scene exists must fail cleanly.
    BgrCamera cam = {{0, 4, 7}, {0, 0.5f, 0}, {0, 1, 0}, 40.f};
    std::vector<uint8_t> rgba(64 * 48 * 4);
    CHECK(bgr_render(r, &cam, 64, 48, 1, 0, rgba.data()) != 0, "render without scene should fail");

    // Out-of-range indices are rejected.
    std::vector<uint32_t> bad = idx;
    bad[0] = 999;
    CHECK(bgr_set_scene(r, pos.data(), nrm.data(), pos.size() / 3, bad.data(), mat.data(), mat.size(), materials, 2) != 0,
          "bad index accepted");

    CHECK(bgr_set_scene(r, pos.data(), nrm.data(), pos.size() / 3, idx.data(), mat.data(), mat.size(), materials, 2) == 0,
          bgr_last_error(r));

    const int w = 160, h = 120;
    std::vector<uint8_t> image(w * h * 4);
    CHECK(bgr_render(r, &cam, w, h, 4, 0, image.data()) == 0, bgr_last_error(r));
    CHECK(bgr_accumulated_samples(r) == 4, "sample count after first render");
    CHECK(bgr_render(r, &cam, w, h, 4, 1, image.data()) == 0, bgr_last_error(r));
    CHECK(bgr_accumulated_samples(r) == 8, "accumulation did not continue");

    // The box (red) should be in the middle of the frame, the floor (green) near the bottom.
    const uint8_t* centre = &image[((h / 2) * w + w / 2) * 4];
    CHECK(centre[0] > centre[1] && centre[0] > 60, "centre pixel is not the red box");
    const uint8_t* bottom = &image[((h - 5) * w + w / 2) * 4];
    CHECK(bottom[1] > bottom[0] && bottom[1] > bottom[2], "bottom pixel is not the green floor");

    bgr_destroy(r);
    std::printf("bgrender smoke test passed\n");
    return 0;
}
