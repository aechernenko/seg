from nnet.bundle.bundle_maker import BundleMaker
from nnet.bundle.sampler.trajectory_based import TrajectoryBasedSampler
from nnet.bundle.sampler.area_based import AreaBasedSampler
from records import train_records, validation_records, test_records
from bundle.c_cache_converter import convert_c_cache


tr_classes = [
    # 'auto_car_move',
    # 'auto_track_move',
    # 'shooting',
    # 'auto-track__move',
    # 'auto-car__move',
    'human__step',
    # 'gate',
    # 'auto_car_asphalt',
    # 'auto-track__digg',
    # 'human__digg',
    'human_step',
    # 'human_digg',
    # 'auto-track__stay',
    # 'auto-car_noise',
    # 'auto-track__digg_ground',
    # 'auto-truck__stay',
    # 'auto-track_digg',
    # 'auto_car__move'
]

ar_classes = [
    'background',
]

# определяем общие правила нарезки
rules = {
    "size_l": 256,
    "size_t": 128,
    "l_thinning": 1,
    "t_thinning": 1,
    "delta_t": 0.3,
    "delta_l": 0.1,
    "max_l_offset": 0.5,
}


if __name__ == '__main__':

    bm = BundleMaker(
        output_path=".",
        name="lim_seg_bundle",
        sample_coords_only=True,
        debug=False,
    )

    bm.set_rules(rules=rules)

    bm.set_records(
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
    )

    bm.add_sampler(name='tr', sampler=TrajectoryBasedSampler, classes=tr_classes)
    # bm.add_sampler(name='ar', sampler=AreaBasedSampler, classes=ar_classes, max_samples_per_cluster=300)

    bm.run()

    convert_c_cache(c_cache='c_cache',
                    bundle_path='../data/c_bundle_hs.h5',
                    train_records=train_records,
                    validation_records=validation_records,
                    test_records=test_records,
                    sample_height=rules['size_t'],
                    sample_width=rules['size_l'],
                    t_thinning=rules['t_thinning'],
                    l_thinning=rules['l_thinning'])
