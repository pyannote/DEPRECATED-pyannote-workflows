# Sciluigi Workflow


## Design convention

When a task expects a `pyannote.core.Annotation` as input, it should only
contain labels of type `string`.

Consequently, when a task outputs a `pyannote.core.Annotation`, it should only
contain labels of type `string`.

## Automatic output

Adding `AutoOutput` to a task will automagically add a `out_put` method to
the task that ...

For the `AutoOutput` mixin to be functional, all inputs must be declared as
task attributes directly.

This will work:

```python
task.in_put1 = task1.out_put
task.in_put2 = task2.out_put
```

This will **not** work:

```python
task.in_put = {
  'input1': task1.out_put
  'input2': task2.out_put
}
```
