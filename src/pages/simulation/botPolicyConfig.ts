export const BOT_POLICY_TYPES = {
    genome: 'genome',
    linearIntent: 'linear_intent',
    random: 'random',
    userControlled: 'user_controlled',
    none: 'none'
} as const;

export type BotPolicyType = typeof BOT_POLICY_TYPES[keyof typeof BOT_POLICY_TYPES];

export type BotPolicyConfig = Record<number, BotPolicyType>;

export function createDefaultBotPolicyConfig(): BotPolicyConfig {
    return {
        1: BOT_POLICY_TYPES.genome,
        2: BOT_POLICY_TYPES.genome,
        3: BOT_POLICY_TYPES.genome,
        4: BOT_POLICY_TYPES.genome
    };
}

export function cloneBotPolicyConfig(config: BotPolicyConfig): BotPolicyConfig {
    return { ...config };
}

export function setBotPolicy(config: BotPolicyConfig, botId: number, policy: BotPolicyType): BotPolicyConfig {
    return {
        ...config,
        [botId]: policy
    };
}

export function setAllBotPolicies(policy: BotPolicyType): BotPolicyConfig {
    return Object.keys(createDefaultBotPolicyConfig()).reduce((nextConfig, botId) => ({
        ...nextConfig,
        [Number(botId)]: policy
    }), {});
}

export function setTeamAPolicy(policy: BotPolicyType): BotPolicyConfig {
    return {
        1: policy,
        2: policy,
        3: BOT_POLICY_TYPES.genome,
        4: BOT_POLICY_TYPES.genome
    };
}

export function setTeamBPolicy(policy: BotPolicyType): BotPolicyConfig {
    return {
        1: BOT_POLICY_TYPES.genome,
        2: BOT_POLICY_TYPES.genome,
        3: policy,
        4: policy
    };
}

export function requiresLinearIntentModel(config: BotPolicyConfig): boolean {
    return Object.values(config).some((policy) => policy === BOT_POLICY_TYPES.linearIntent);
}

export function formatBotPolicy(policy: BotPolicyType): string {
    if (policy === BOT_POLICY_TYPES.linearIntent) return 'Linear Intent';
    if (policy === BOT_POLICY_TYPES.random) return 'Random';
    if (policy === BOT_POLICY_TYPES.userControlled) return 'User Controlled';
    if (policy === BOT_POLICY_TYPES.none) return 'None';
    return 'Genome';
}
