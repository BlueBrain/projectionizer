Frequently Asked Questions
==========================

 .. _FAQ_Apron:

Apron
-----

**Well, what about apron? What is this apron you speak of and what is the magic therein?**

In case of hex columns, there are so-called edge effects measurable in terms of neuron count per fiber as can clearly be seen in the image below. This effect happens due to there not being synapses for the fibers to connect to on the other side of the edge of the region.

.. image:: images/edges_no_apron.png
   :height: 300

This is where the apron comes in. Apron is a rectangular box limited by XZ coordinates (Y-axis-wise it extends from -infinity to infinity) that surrounds the column.
When apron is used, Projectionizer will add synapses inside the bounding box (i.e., also outside the region) and thus mitigating the edge effect as can be seen in the image below.

.. image:: images/edges_apron.png
   :height: 300

Configuration File
------------------

**I am used to running projectionizer with the old config.
I really do not want to read the whole** :ref:`configuration` **section again.
How do I get my projections?**

Fortunately, there is an :ref:`example<Config_ExampleFile>` one can check and compare the old configuration to it.
It is *mostly* the same as the old one: some parameters might have changed place or name and some are added or removed.
These changes should also be reflected in the :ref:`changelog`.

Sbatch Issues
-------------

**Oh man, I'm trying to run projectionizer but it seems that the spykfunc phase has issues. Can I take the rest of the day off and leave you with the issue?**

Unfortunately, not, but we'll help you with the issue.

`spykfunc` is a delicate beast.
First of all, make sure that you run projectionizer in an exclusive node.
Secondly, if you are running it with sbatch, make sure that you set ``--ntasks-per-node=1`` since, otherwise, spykfunc might launch multiple schedulers and crash.
Thirdly, make sure that you use ``--constraint=nvme`` for the allocation, since spykfunc needs to access the NVMe drives.

If you still have issues, the minions at NSE have forged an example :ref:`sbatch script<Index_ExampleSbatch>` that you can use.

 .. _FAQ_Indexing:

0-based vs 1-based indexing
---------------------------

**I'm totally lost here. In which case the GID indexing is 0-based, in which is it 1-based? Why the difference?**

Yeah, we totally feel you. We've burnt our fingers trying to manage this issue more often than we'd care to admit.

As to why do we have this issue in first place: Historically, at BBP, we have used 1-based indexing for source and target cell IDs (SGID, TGID) whereas SONATA format relies on 0-based indexing.
Naturally, projectionizer used to conform to the 1-based indexing of the mvd3/bluepy/BlueConfig file formats, too.

As we gradually started switching to the newer SONATA format in BBP, also projectionizer needed to take this into account.
However, some of the packages (e.g., libFLATIndex) used by projectionizer were still using 1-based indexing.
Hence, there was a period of time in which projectionizer had to simultaneously take both formats into account.
This resulted in minor hiccups (read: total chaos) while trying to make sense which GID refers to which and moreover, in which file.

Starting from projectionizer **v3.0.0**, this is no longer the case: now everything but the `user.target` files is conforming to the 0-based indexing.
In the table below, we have listed which GID indexing the output files conform to in different projectionizer versions.

.. table::

  +------------+------------+------------+----------------------------------------------+
  | File Type  | GID indexing /          | Further explanation                          |
  |            | projectionizer version  |                                              |
  +            +------------+------------+                                              +
  |            |  < 3.0.0   | >= 3.0.0   |                                              |
  +============+============+============+==============================================+
  | CSV        | 1-based    | 0-based    | SGID of the virtual fibers.                  |
  +------------+------------+------------+----------------------------------------------+
  | feather    | 1-based    | 0-based    | SGID / TGID in the various computed states.  |
  +------------+------------+------------+----------------------------------------------+
  | h5         | 0-based    | 0-based    | SGID / TGID (node ids) in the SONATA files.  |
  +------------+------------+------------+----------------------------------------------+
  | user.target| 1-based    | 1-based    | SGID of the virtual fibers.                  |
  |            |            |            |                                              |
  |            |            |            | - used in BlueConfig (e.g., for simulation)  |
  |            |            |            | - not used by SONATA                         |
  |            |            |            | - will eventually be removed                 |
  +------------+------------+------------+----------------------------------------------+


Getting Help
------------

**I have read the documentation, I still can't get my projections to run. HALP!**

Worry not. Just summon us using the call sign and we'll be there for you:

.. image:: images/halp.png
   :height: 300
