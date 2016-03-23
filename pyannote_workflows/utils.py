import luigi
import sciluigi

import six
import json
import hashlib


class Hyperopt(sciluigi.Task):

    temp = luigi.Parameter()
    in_evaluation = None

    def out_put(self):
        return sciluigi.TargetInfo(self, self.temp)

    def run(self):
        with self.out_put().open('w') as fp:
            fp.write(self.in_evaluation().path)


class AutoOutput(object):

    def _output_from_hash(self):

        # working directory within which all automatic outputs will be stored
        workdir = self.workflow_task.workdir

        description = {}

        # add one {key: value} per in_xxxx method
        # key = 'in_xxxx'
        # value = F(in_xxxx().path)
        for attrname, attrval in six.iteritems(self.__dict__):
            if 'in_' == attrname[0:3]:
                path = attrval().path
                if path.startswith(workdir):
                    path = path[len(workdir):]
                description[attrname] = path

        # add one {key: value} per task parameter
        # key = parameter name
        # value = parameter value
        params = self.get_params()
        params = [name for name, _ in params]
        for param_name in params:
            # do not take 'instance_name' and 'workflow_task' into account
            if param_name in ['instance_name', 'workflow_task']:
                continue
            description[param_name] = getattr(self, param_name)

        # hash the resulting dictionary
        digest = hashlib.sha1(
            json.dumps(description, sort_keys=True)).hexdigest()

        # generate out_put path automatically
        output_path = '{workdir}/{workflow_name}/{instance_name}/{digest}'
        return output_path.format(
            workdir=workdir,
            instance_name=self.instance_name,
            workflow_name=self.workflow_task.__class__.__name__,
            digest=digest)

    def out_put(self):
        # automagically get out_put path
        path = self._output_from_hash()
        return sciluigi.TargetInfo(self, path)


def getAllOutputs(workflow):
    out_puts = {}
    for instance_name, task in six.iteritems(workflow._tasks):
        out_puts[instance_name] = task.out_put().path
    return out_puts

# def hyperoptify(Workflow, output='out_put'):
#
#     # add 'hyperoptified' paramater
#     Workflow.hyperoptified = luigi.Parameter()
#
#     original_workflow_method = Workflow.workflow
#
#     def new_workflow_method(self):
#
#         print 'hyperoptified = ' + self.hyperoptified
#
#         final_task = original_workflow_method(self)
#
#         hyperopt = self.new_task(
#             'hyperopt', Hyperopt, temp=self.hyperoptified)
#         hyperopt.in_final = getattr(final_task, output)
#
#         return hyperopt
#
#     Workflow.workflow = new_workflow_method
#
#     return Workflow
#
#
# class HyperoptMixin(object):
#
#     hyperoptified = luigi.Parameter()
#
#     def workflow(self):
#
#         final_task = super(self, HyperoptMixin).worflow()
#
#         hyperopt = self.new_task(
#             'hyperopt', Hyperopt, temp=self.hyperoptified)
#         hyperopt.in_final = getattr(final_task, 'out_put')
#
#         return hyperopt
