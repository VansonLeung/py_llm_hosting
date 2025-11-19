const { Sequelize, DataTypes } = require('sequelize');
const path = require('path');

// Initialize Sequelize with SQLite
const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: path.join(__dirname, 'database.sqlite'),
  logging: false
});

// Define Conversation Model
const Conversation = sequelize.define('Conversation', {
  id: {
    type: DataTypes.UUID,
    defaultValue: DataTypes.UUIDV4,
    primaryKey: true
  },
  title: {
    type: DataTypes.STRING,
    allowNull: false,
    defaultValue: 'New conversation'
  },
  endpointId: {
    type: DataTypes.STRING,
    allowNull: true
  },
  model: {
    type: DataTypes.STRING,
    defaultValue: 'gpt-4o-mini'
  },
  toolIds: {
    type: DataTypes.JSON,
    defaultValue: []
  },
  mcpToolIds: {
    type: DataTypes.JSON,
    defaultValue: []
  },
  tokenUsage: {
    type: DataTypes.JSON,
    defaultValue: { prompt: 0, completion: 0, total: 0 }
  }
}, {
  timestamps: true
});

// Define Message Model
const Message = sequelize.define('Message', {
  id: {
    type: DataTypes.UUID,
    defaultValue: DataTypes.UUIDV4,
    primaryKey: true
  },
  role: {
    type: DataTypes.STRING,
    allowNull: false
  },
  content: {
    type: DataTypes.JSON, // Store as JSON to handle mixed content (text/arrays)
    allowNull: false
  },
  attachments: {
    type: DataTypes.JSON,
    defaultValue: []
  },
  metadata: {
    type: DataTypes.JSON,
    defaultValue: {}
  },
  tokenUsage: {
    type: DataTypes.JSON,
    defaultValue: null
  }
}, {
  timestamps: true
});

// Define Relationships
Conversation.hasMany(Message, { as: 'messages', foreignKey: 'conversationId', onDelete: 'CASCADE' });
Message.belongsTo(Conversation, { foreignKey: 'conversationId' });

module.exports = {
  sequelize,
  Conversation,
  Message
};
