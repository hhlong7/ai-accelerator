"""
This is for the discrete event simulator engine. It doesnt step thru time cycle by cycle
but jumps from and event to the next one for eg like kernel 3 arrived at t = 100s => kernel 2 arrived
at t = 300s => etc. these events are stored in a priority queue ordered by time => the engine takes the
one that is the fastest to compute and run the code for it.
"""

"""
so this means that:
- the event would be in time order
- the ones with the same time would be in order of when they were added to the queue
- they can be cancelled if not need anymore
"""

import heapq

queued = "queued"
finished = "finished"

class engine:
    def __init__(self):
        self.event_queue = [] #use heap
        self.current_time = 0.00
        self.count_event = 0 
        self.handlers = {}
        self.cancelled_e = set()

    #reg handler for event type to call handler(engine, data)
    def register(self, event_type, handler):
        self.handlers[event_type] = handler

    #schedule an event to da queue, return the id so if we want to cancel we can use that
    def schedule(self, time, type, data=None):
        if time < self.current_time:
            raise ValueError(f"Cant schedule an event {type} at t = {time} since current time is {self.current_time}")
        id_seq = self.count_event
        heapq.heappush(self.event_queue, (time, id_seq, type, data))
        self.count_event += 1
        return id_seq

    #cancel an event
    def cancel(self, id):
        self.cancelled_e.add(id)

    #run the engine in time order till queue is empty or next event > stop, return final time
    def main(self, stop=None):
        while self.event_queue:
            if stop is not None and self.event_queue[0][0] > stop:
                break
            time, id_seq, type, data = heapq.heappop(self.event_queue)

            #cancelled event are not processed with the current time so clk dont increment for cancelled events
            if id_seq in self.cancelled_e:
                self.cancelled_e.discard(id_seq)
                continue

            self.current_time = time
            handler = self.handlers.get(type)
            if handler is None:
                raise KeyError(f"There're no handler registered for the event type {type}")
            handler(self, data)

        return self.current_time
