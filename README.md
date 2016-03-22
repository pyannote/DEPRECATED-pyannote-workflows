# Sciluigi Workflow


## Design convention

When a task expects a `pyannote.core.Annotation` as input, it should only
contain labels of type `string`.

Consequently, when a task outputs a `pyannote.core.Annotation`, it should only
contain labels of type `string`.

## Automatic output

Adding `AutoOutput` mixin to a `sciluigi.Task` will automagically add method
called `out_put` that generates a file whose path is derived from the hashing
or all `in_xxxx().path` and all task parameters with the following structure:

```
workdir/workflow_name/instance_name/hash
```

where
  * `workdir` is the value of parameter `sciluigi.WorkflowTask.workdir`,
  * `workflow_name` is the workflow class name,
  * `instance_name` is the name of the task used in `self.new_task(instance_name, ...)`
  * `hash` is a hash that uniquely identify a task based on its inputs paths
  and its parameters.

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
