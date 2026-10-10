import pytest

from sim.engine import queued, finished, Engine

def log_event():
    
    #logging when the queued and finished handler just record time, data
    engine = Engine()
    log = []
    def record(eng, data):
        log.append((eng.current_time, data))

    engine.register(queued, record)
    engine.register(finished, record)
    return engine, log

#only for in time order, not scheduled order
def test_events_in_time_order():
    engine, log = log_event()
    events = [(5.00, "e"), (1.00, "a"), (2.00, "b"), (1.50, "c"), (3.00, "d")]
    for time, data in events:
        engine.schedule(time, queued, data)
    end = engine.main() #return end time
    assert log == [(1.00, "a"), (1.50, "c"), (2.00, "b"), (3.00, "d"), (5.00, "e")]
    assert end == 5.00


#this time in order of when its scheduled
def test_events_in_scheduled_order_samet():
    engine, log = log_event()
    engine.schedule(1.00, finished, "first")
    engine.schedule(1.00, queued, "second")
    engine.schedule(1.00, finished, "third")
    engine.schedule(1.00, queued, "fourth")
    
    end = engine.main() #return end time
    assert [e for _, e in log] == ["first", "second", "third", "fourth"]

#tie in time shouldnt compare the data, dicts cant be compared with < so this crash without id_seq
def test_tie_doesnt_compare_data():
    engine, log = log_event()
    engine.schedule(1.00, queued, {"id": 1})
    engine.schedule(1.00, queued, {"id": 2})

    engine.main()
    assert [e["id"] for _, e in log] == [1, 2]


#handler can schedule new events in the future, like queued => finished for a kernel
def test_handler_schedule_future_events():
    engine = Engine()
    log = []

    def on_queued(eng, kernel):
        log.append(("queued", eng.current_time, kernel))
        eng.schedule(eng.current_time + 10.00, finished, kernel)

    def on_finished(eng, kernel):
        log.append(("finished", eng.current_time, kernel))

    engine.register(queued, on_queued)
    engine.register(finished, on_finished)
    engine.schedule(0.00, queued, "k0")
    engine.schedule(3.00, queued, "k1")

    engine.main()
    assert log == [
        ("queued", 0.00, "k0"),
        ("queued", 3.00, "k1"),
        ("finished", 10.00, "k0"),
        ("finished", 13.00, "k1"),
    ]


#cancelled event never run and the clk dont move to its time
def test_cancelled_event_not_run():
    engine, log = log_event()
    engine.schedule(1.00, queued, "keep")
    cancel_id = engine.schedule(9.00, finished, "cancel this")
    engine.cancel(cancel_id)

    end = engine.main() #return end time
    assert log == [(1.00, "keep")]
    assert end == 1.00


#step 1.4 preview: kernel finish time changes => cancel old finished event and schedule new one
def test_cancel_and_reschedule():
    engine, log = log_event()
    old_finish = engine.schedule(10.00, finished, "k0")

    def on_queued(eng, data):
        eng.cancel(old_finish)
        eng.schedule(15.00, finished, "k0")

    engine.register(queued, on_queued) #replace the record handler for queued
    engine.schedule(4.00, queued, None)

    engine.main()
    assert log == [(15.00, "k0")]


#schedule before current time should raise error
def test_schedule_in_past_raise():
    engine = Engine()

    def on_queued(eng, data):
        eng.schedule(eng.current_time - 1.00, finished, None)

    engine.register(queued, on_queued)
    engine.schedule(5.00, queued, None)

    with pytest.raises(ValueError):
        engine.main()


#event type with no handler registered should raise error
def test_no_handler_raise():
    engine = Engine()
    engine.schedule(1.00, "random", None)

    with pytest.raises(KeyError):
        engine.main()


#stop at the stop time, then can continue running after
def test_main_stop_and_continue():
    engine, log = log_event()
    for time in [1.00, 2.00, 3.00, 4.00]:
        engine.schedule(time, queued, time)

    engine.main(stop=2.50)
    assert [e for _, e in log] == [1.00, 2.00]
    assert engine.current_time == 2.00

    engine.main() #continue till queue is empty
    assert [e for _, e in log] == [1.00, 2.00, 3.00, 4.00]