"""«Мир или гитара»: мир надоел - садится играть, музыка надоела - встаёт; гистерезис."""
import pytest

from music.mood import MIN_ATTEMPTS, MIN_WORLD, REF_ERR, Mood


def walk(mood, err, seconds, dt=0.5):
    for _ in range(int(seconds / dt)):
        mood.world_tick(err, dt)


def test_a_fresh_world_keeps_him_walking():
    m = Mood("sometimes")
    walk(m, REF_ERR, 120)
    assert not m.want_play()


def test_a_bored_world_makes_him_play():
    m = Mood("sometimes")
    walk(m, REF_ERR * 0.1, 60)
    assert m.want_play()


def test_at_least_twenty_seconds_in_the_world_between_sessions():
    m = Mood("often")
    walk(m, 0.0, MIN_WORLD - 1)
    assert not m.want_play()
    walk(m, 0.0, 2)
    assert m.want_play()
    m.start()
    for _ in range(MIN_ATTEMPTS):
        m.after_attempt(novelty=0.0, progress=0.0)
    assert m.want_stop()
    m.stop()
    walk(m, 0.0, MIN_WORLD - 1)
    assert not m.want_play()


def test_boring_music_makes_him_get_up_but_not_before_three_attempts():
    m = Mood("sometimes")
    walk(m, 0.0, 60)
    m.start()
    m.after_attempt(novelty=0.0, progress=0.0)
    m.after_attempt(novelty=0.0, progress=0.0)
    assert not m.want_stop()                                # меньше трёх попыток
    m.after_attempt(novelty=0.0, progress=0.0)
    assert m.want_stop()


def test_interesting_music_keeps_him_playing():
    m = Mood("sometimes")
    walk(m, 0.0, 60)
    m.start()
    for _ in range(10):
        m.after_attempt(novelty=0.9, progress=0.9)
    assert not m.want_stop()


@pytest.mark.parametrize("err_share", [0.3, 0.42])
def test_how_often_setting_shifts_the_threshold(err_share):
    decisions = {}
    for freq in ("rare", "sometimes", "often"):
        m = Mood(freq)
        walk(m, REF_ERR * err_share * 2, 90)
        decisions[freq] = m.want_play()
    order = ["rare", "sometimes", "often"]
    # кто играет при «редко», тот играет и при более частых
    for a, b in zip(order, order[1:]):
        assert decisions[a] <= decisions[b]
    assert decisions["often"] and not decisions["rare"]


def test_mood_state_survives_as_dict():
    m = Mood("often")
    walk(m, 0.0, 30)
    again = Mood.from_dict(m.to_dict(), "often")
    assert again.world == pytest.approx(m.world) and again.music == pytest.approx(m.music)
