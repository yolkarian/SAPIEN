.. _contact:

Contact
==================

.. highlight:: python

Contact information is useful for detecting collisions, grasps, and impacts.
Current SAPIEN contacts expose PhysX components and collision shapes rather than
old ``actor0``/``actor1`` fields.

In this tutorial, you will learn how to read ``PhysxContact`` and
``PhysxContactPoint`` objects.

Minimal example
----------------

.. code-block:: python

   import sapien

   scene = sapien.Scene()
   scene.add_ground(0)

   builder = scene.create_actor_builder()
   builder.add_box_collision(half_size=[0.2, 0.2, 0.2])
   builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[1, 0, 0])
   box = builder.build(name="box")
   box.set_pose(sapien.Pose([0, 0, 1]))

   for _ in range(120):
      scene.step()

   for contact in scene.get_contacts():
      body0, body1 = contact.bodies
      shape0, shape1 = contact.shapes
      entity0 = body0.entity
      entity1 = body1.entity
      print("contact:", entity0.name, entity1.name, shape0, shape1)

      for point in contact.points:
         print("position", point.position)
         print("normal", point.normal)
         print("impulse", point.impulse)
         print("separation", point.separation)

Contact fields
---------------

``scene.get_contacts()`` returns a list of ``sapien.physx.PhysxContact``.

* ``contact.bodies``: two ``PhysxRigidBaseComponent`` objects involved in the
  contact. Use ``body.entity`` to get their owning entities.
* ``contact.shapes``: two ``PhysxCollisionShape`` objects involved in the
  contact.
* ``contact.points``: a list of contact points.

For each ``PhysxContactPoint``:

* ``impulse`` is the impulse vector applied by the solver;
* ``normal`` is the contact normal;
* ``position`` is the world-space contact position;
* ``separation`` is the signed separation distance for the contact pair.

.. note::

   A contact object can be generated while contact is beginning or ending, not
   only while two shapes are visibly resting on each other.
