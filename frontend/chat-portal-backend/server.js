const express = require('express');
const cors = require('cors');
const { sequelize, Conversation, Message } = require('./models');

const app = express();
const PORT = process.env.PORT || 5278;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased limit for attachments

// --- Conversations API ---

// List all conversations (summary only, no messages)
app.get('/api/conversations', async (req, res) => {
  try {
    const conversations = await Conversation.findAll({
      order: [['updatedAt', 'DESC']]
    });
    res.json(conversations);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get single conversation with messages
app.get('/api/conversations/:id', async (req, res) => {
  try {
    const conversation = await Conversation.findByPk(req.params.id, {
      include: [{
        model: Message,
        as: 'messages',
        // Order messages by creation time
        // Note: In a real app, you might want pagination here
      }],
      order: [
        [{ model: Message, as: 'messages' }, 'createdAt', 'ASC']
      ]
    });
    
    if (!conversation) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    res.json(conversation);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Create new conversation
app.post('/api/conversations', async (req, res) => {
  try {
    const conversation = await Conversation.create(req.body);
    // Return with empty messages array to match expected format
    const result = conversation.toJSON();
    result.messages = [];
    res.status(201).json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Update conversation
app.patch('/api/conversations/:id', async (req, res) => {
  try {
    const [updated] = await Conversation.update(req.body, {
      where: { id: req.params.id }
    });
    
    if (!updated) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    const conversation = await Conversation.findByPk(req.params.id);
    res.json(conversation);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Delete conversation
app.delete('/api/conversations/:id', async (req, res) => {
  try {
    const deleted = await Conversation.destroy({
      where: { id: req.params.id }
    });
    
    if (!deleted) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    res.status(204).send();
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// --- Messages API ---

// Add message to conversation
app.post('/api/conversations/:id/messages', async (req, res) => {
  try {
    const conversationId = req.params.id;
    const conversation = await Conversation.findByPk(conversationId);
    
    if (!conversation) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    const message = await Message.create({
      ...req.body,
      conversationId
    });
    
    // Update conversation timestamp
    await conversation.changed('updatedAt', true);
    await conversation.save();
    
    res.status(201).json(message);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Update message (e.g. for streaming updates)
app.patch('/api/messages/:id', async (req, res) => {
  try {
    const [updated] = await Message.update(req.body, {
      where: { id: req.params.id }
    });
    
    if (!updated) {
      return res.status(404).json({ error: 'Message not found' });
    }
    
    const message = await Message.findByPk(req.params.id);
    
    // Also update parent conversation timestamp
    const conversation = await Conversation.findByPk(message.conversationId);
    if (conversation) {
      await conversation.changed('updatedAt', true);
      await conversation.save();
    }
    
    res.json(message);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Start server
async function startServer() {
  try {
    await sequelize.sync(); // Sync database models
    console.log('Database synced successfully');
    
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error('Unable to start server:', error);
  }
}

if (require.main === module) {
  startServer();
}

module.exports = app;
